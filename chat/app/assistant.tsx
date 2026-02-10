"use client";

import { AssistantRuntimeProvider, useLocalRuntime, type ChatModelAdapter } from "@assistant-ui/react";
import { Thread } from "@/components/assistant-ui/thread";

const MyModelAdapter: ChatModelAdapter = {
  async *run({ messages }) {
    const lastMessage = messages[messages.length - 1];
    const query = lastMessage?.role === "user" 
      ? lastMessage.content.find((p) => p.type === "text")?.text || ""
      : "";

    if (!query) return;

    const response = await fetch("/api/chat", {
      method: "POST",
      body: JSON.stringify({ 
        messages: messages.map(m => ({
          role: m.role,
          content: m.content
        })) 
      }),
      headers: { "Content-Type": "application/json" },
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch: ${response.statusText}`);
    }

    const reader = response.body?.getReader();
    if (!reader) return;

    const decoder = new TextDecoder();
    let content = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      content += chunk;

      yield {
        content: [{ type: "text", text: content }],
      };
    }
  },
};

export const Assistant = () => {
  const runtime = useLocalRuntime(MyModelAdapter);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="h-dvh">
        <Thread />
      </div>
    </AssistantRuntimeProvider>
  );
};
