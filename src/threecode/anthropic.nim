## Native Anthropic Messages API (`/v1/messages`) request building.
##
## 3code's pipeline is OpenAI-shaped end to end: history is stored as
## `{role, content, tool_calls}` / `{role:"tool", tool_call_id, content}`, and
## tools as `{type:"function", function:{name, description, parameters}}`. The
## native Claude API wants a different shape — a top-level `system`, `tool_use`
## / `tool_result` content blocks, `input_schema` tools — so this module
## translates the OpenAI-shaped request into the Anthropic shape on the way
## out. The SSE response is translated back into the OpenAI-shaped assistant
## message inside `api.streamHttp`, so the rest of the pipeline never sees the
## difference.
##
## Why native instead of the OpenAI-compatible endpoint: prompt caching. The
## compat layer disables it; the native API supports `cache_control`
## breakpoints, which cut the cost of long repo sessions (the stable system
## prompt + tool schemas + growing history are re-read from cache at ~0.1x
## instead of re-billed in full each turn).

import std/[json, strutils]
import types, prompts

const AnthropicVersion* = "2023-06-01"

proc cacheControl(): JsonNode =
  ## A `cache_control` breakpoint at the configured TTL. 1h needs no beta
  ## header (verified against the live API).
  if cacheOneHour: %*{"type": "ephemeral", "ttl": "1h"}
  else: %*{"type": "ephemeral"}

proc oaiToolsToAnthropic(tools: JsonNode): JsonNode =
  ## `{type:"function", function:{name, description, parameters}}` →
  ## `{name, description, input_schema}`.
  result = newJArray()
  if tools == nil or tools.kind != JArray: return
  for t in tools:
    let fn = t{"function"}
    if fn == nil or fn.kind != JObject: continue
    let schema =
      if fn{"parameters"} != nil and fn{"parameters"}.kind == JObject: fn{"parameters"}
      else: %*{"type": "object", "properties": newJObject()}
    result.add %*{
      "name": fn{"name"}.getStr,
      "description": fn{"description"}.getStr,
      "input_schema": schema,
    }

proc parseToolInput(arguments: string): JsonNode =
  ## OpenAI tool args are a JSON *string*; Anthropic `tool_use.input` is a JSON
  ## *object*. Empty / malformed args become `{}` (a tool call already in
  ## history was accepted once, so this is belt-and-suspenders).
  let s = arguments.strip
  if s.len == 0: return newJObject()
  try:
    let j = parseJson(s)
    if j.kind == JObject: j else: newJObject()
  except CatchableError:
    newJObject()

proc toAnthropicMessages(messages: JsonNode): tuple[system: string, msgs: JsonNode] =
  ## Translate the OpenAI-shaped history into a top-level system string plus an
  ## Anthropic `messages` array (all messages carry block-array content so
  ## cache_control placement is uniform). Consecutive `tool` messages are
  ## coalesced into one `user` turn of `tool_result` blocks — Anthropic wants
  ## every parallel tool result in a single user message. Unknown bookkeeping
  ## fields (`usage`, `interrupted`, `reasoning_content`) are simply not read.
  result.msgs = newJArray()
  var systemParts: seq[string]
  var lastWasToolResult = false
  for m in messages:
    if m.kind != JObject: continue
    let role = m{"role"}.getStr
    case role
    of "system":
      let c = m{"content"}.getStr
      if c.len > 0: systemParts.add c
      lastWasToolResult = false
    of "tool":
      let tr = %*{
        "type": "tool_result",
        "tool_use_id": m{"tool_call_id"}.getStr,
        "content": m{"content"}.getStr,
      }
      if lastWasToolResult and result.msgs.len > 0:
        result.msgs[^1]["content"].add tr
      else:
        result.msgs.add %*{"role": "user", "content": [tr]}
      lastWasToolResult = true
    of "user":
      result.msgs.add %*{
        "role": "user",
        "content": [%*{"type": "text", "text": m{"content"}.getStr}],
      }
      lastWasToolResult = false
    of "assistant":
      var blocks = newJArray()
      let text = m{"content"}.getStr
      if text.len > 0:
        blocks.add %*{"type": "text", "text": text}
      let tcs = m{"tool_calls"}
      if tcs != nil and tcs.kind == JArray:
        for tc in tcs:
          let fn = tc{"function"}
          if fn == nil: continue
          blocks.add %*{
            "type": "tool_use",
            "id": tc{"id"}.getStr,
            "name": fn{"name"}.getStr,
            "input": parseToolInput(fn{"arguments"}.getStr),
          }
      # An assistant turn must have at least one block.
      if blocks.len == 0:
        blocks.add %*{"type": "text", "text": ""}
      result.msgs.add %*{"role": "assistant", "content": blocks}
      lastWasToolResult = false
    else:
      lastWasToolResult = false
  result.system = systemParts.join("\n")

proc buildAnthropicBody*(p: Profile, messages: JsonNode): JsonNode =
  ## Assemble a `/v1/messages` request body from the OpenAI-shaped history.
  ## Two `cache_control` breakpoints: one on the system block (caches the tool
  ## schemas + system prompt, which render before it), one on the last message
  ## block (caches the conversation prefix for the next turn). No `thinking` /
  ## `output_config` for now — Claude runs without extended thinking, which is
  ## the cheap default and sidesteps thinking-block replay; the `:reasoning`
  ## knob is therefore inert on the native path until that lands.
  let (system, msgs) = toAnthropicMessages(messages)
  let gen = knownGoodGeneration(p)
  let maxTok = if gen.maxTokens > 0: gen.maxTokens else: 4096
  result = %*{
    "model": p.model,
    "max_tokens": maxTok,
    "stream": true,
    "messages": msgs,
  }
  var sysBlock = %*{"type": "text", "text": system}
  sysBlock["cache_control"] = cacheControl()
  result["system"] = %[sysBlock]
  let tools = oaiToolsToAnthropic(setup(p).tools)
  if tools.len > 0:
    result["tools"] = tools
  if msgs.len > 0:
    let lastContent = msgs[^1]{"content"}
    if lastContent != nil and lastContent.kind == JArray and lastContent.len > 0:
      lastContent[^1]["cache_control"] = cacheControl()

proc buildAnthropicSummaryBody*(p: Profile, messages: JsonNode,
                                systemPrompt: string, maxTokens: int): JsonNode =
  ## Non-streaming `/v1/messages` body for the meta-summarizer: a dedicated
  ## summarizer system prompt over the translated conversation, no tools and
  ## no cache_control (it's a one-off call). The session's own system message
  ## is dropped by `toAnthropicMessages`.
  ##
  ## A resumed transcript usually ends on an assistant turn; Anthropic only
  ## generates when the last message is `user`, so append an explicit directive
  ## to trigger the recap (coalesced into a trailing user turn if there is one).
  let (_, msgs) = toAnthropicMessages(messages)
  let directive = %*{"type": "text",
    "text": "Summarize the conversation above into a recap, following your system instructions."}
  if msgs.len > 0 and msgs[^1]{"role"}.getStr == "user":
    msgs[^1]["content"].add directive
  else:
    msgs.add %*{"role": "user", "content": [directive]}
  result = %*{
    "model": p.model,
    "max_tokens": maxTokens,
    "stream": false,
    "system": systemPrompt,
    "messages": msgs,
  }
