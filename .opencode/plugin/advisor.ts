import { Plugin } from "@opencode/plugin";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";
import type { AdvisorConfig } from "../../src/config";
import { DEFAULT_ADVISOR_CONFIG, parseAdvisorConfig, parseAdvisorConfigJSON } from "../../src/config";
import { parseModelID } from "../../src/model_id";
import { RateLimiter } from "../../src/rate_limiter";
import { extractTranscriptThenTruncate, type TranscriptMessage } from "../../src/transcript";
import { buildAdvisorPrompt } from "../../src/prompt";

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readConfigFile(directory: string): AdvisorConfig {
  const candidates = [
    join(directory, ".opencode", "oc-advisor.json"),
    join(homedir(), ".config", "opencode", "oc-advisor.json"),
  ];
  for (const path of candidates) {
    try {
      const text = readFileSync(path, "utf-8");
      return parseAdvisorConfigJSON(text);
    } catch {
      // try next
    }
  }
  return DEFAULT_ADVISOR_CONFIG;
}

// V2 session messages are flat unions: user messages carry `text`, assistant
// messages carry a `content` array of {type: "text" | "reasoning" | "tool", ...}.
function toTranscript(messages: ReadonlyArray<unknown>): ReadonlyArray<TranscriptMessage> {
  return messages
    .map((m): TranscriptMessage | undefined => {
      const msg = m as Record<string, unknown> | undefined;
      if (msg === null || typeof msg !== "object") return undefined;
      const role = typeof msg.type === "string" ? msg.type : "unknown";
      let text = "";
      if (typeof msg.text === "string") {
        text = msg.text;
      } else if (Array.isArray(msg.content)) {
        text = msg.content
          .map((p) => (p as Record<string, unknown> | undefined))
          .filter((p): p is { type: "text"; text: string } => p !== null && typeof p === "object" && p.type === "text" && typeof p.text === "string")
          .map((p) => p.text)
          .join(" ");
      }
      const trimmed = text.trim();
      return trimmed.length > 0 ? { info: { role }, parts: [{ type: "text", text: trimmed }] } : undefined;
    })
    .filter((m): m is TranscriptMessage => m !== undefined);
}

export default Plugin.define({
  id: "oc-advisor",
  async setup(ctx) {
    const hasOptions = isPlainObject(ctx.options) && Object.keys(ctx.options).length > 0;
    const config: AdvisorConfig = hasOptions
      ? parseAdvisorConfig(ctx.options)
      : readConfigFile(ctx.location.directory);
    const { providerID, modelID } = parseModelID(config.advisorModel);
    const limiter = new RateLimiter(config.maxCallsPerSession);

    await ctx.tool.transform((editor) => {
      editor.add({
        name: "ask_advisor",
        description:
          "Consult a senior advisor model for strategic guidance, architecture review, a second opinion, or a course correction when you are stuck or unsure. Use this sparingly, only when a non-obvious design decision or hidden risk is in play. The advisor runs on a separate, higher-intelligence model and receives the current session transcript plus your question.",
        input: {
          type: "object",
          properties: {
            question: {
              type: "string",
              description: "The specific question or decision you want advice on. Be concrete.",
            },
            context: {
              type: "string",
              description:
                "Optional extra context: file paths, code snippets, prior decisions, or constraints the advisor should weigh.",
            },
          },
          required: ["question"],
          additionalProperties: false,
        },
        async execute(input, toolCtx) {
          const args = input as { question?: string; context?: string };
          const question = typeof args.question === "string" ? args.question : "";
          const extraContext = typeof args.context === "string" ? args.context : undefined;
          if (question.length === 0) {
            return {
              content: "ask_advisor requires a non-empty 'question'.",
            };
          }

          const acquired = limiter.tryAcquire(toolCtx.sessionID);
          if (!acquired.allowed) {
            return {
              content: `Advisor call limit (${config.maxCallsPerSession}) reached for this session. Raise "maxCallsPerSession" in .opencode/oc-advisor.json, or start a new session.`,
            };
          }

          try {
            const messages = await ctx.session.context({ sessionID: toolCtx.sessionID });
            const transcript = extractTranscriptThenTruncate(
              toTranscript(messages as ReadonlyArray<unknown>),
              config.transcriptCharBudget,
            );
            const { system, user } = buildAdvisorPrompt({
              transcript,
              question,
              context: extraContext,
            });

            const result = await ctx.experimental.generate.text({
              model: { providerID, id: modelID },
              prompt: `${system}\n\n---\n\n${user}`,
            });

            limiter.commit(toolCtx.sessionID);
            const text = result.text.trim();
            return {
              content: text.length > 0 ? text : "(advisor returned no text response)",
            };
          } catch (error) {
            limiter.refund(toolCtx.sessionID);
            return {
              content: `Advisor model (${config.advisorModel}) failed: ${String(error)}`,
            };
          }
        },
      });
    });
  },
});
