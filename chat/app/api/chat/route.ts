export async function POST(req: Request) {
  try {
    const { messages }: { messages: any[] } = await req.json();
    
    // Find the last user message to use as the primary prompt
    const lastUserMessage = [...messages].reverse().find((m) => m.role === "user");
    
    // Extract the text from the content parts if it's an array
    let query = "";
    if (lastUserMessage) {
      if (Array.isArray(lastUserMessage.content)) {
        query = lastUserMessage.content.find((p: any) => p.type === "text")?.text || "";
      } else {
        query = lastUserMessage.content || "";
      }
    }

    const ragResponse = await fetch(process.env.RAG_API_URL || 'http://localhost:8000/stream-rag', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.RAG_API_KEY}`
      },
      body: JSON.stringify({ 
        prompt: query, 
        messages: messages, // Send full history for context
        top_k: 5 
      }),
    });

    if (!ragResponse.ok) {
      return new Response(`RAG service error: ${ragResponse.statusText}`, { status: ragResponse.status });
    }

    // Return the raw stream directly to the client
    return new Response(ragResponse.body, {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
    });
  } catch (error) {
    console.error("API Route Error:", error);
    return new Response("Internal Server Error", { status: 500 });
  }
}
