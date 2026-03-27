import React, { useState, useEffect, useRef, useCallback } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8000";

// ─── Types ────────────────────────────────────────────────────────────────────

interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: string;
  modulesActivated?: string[];
  confidence?: number;
}

interface MemoryStatus {
  working_memory: { active_sessions: number; total_tokens: number };
  episodic_memory: { initialised: boolean; entry_count: number };
  knowledge_base: { initialised: boolean; fact_count: number; graph_nodes: number };
}

interface SystemHealth {
  status: string;
  cee: { running: boolean; time_step: number; uptime_seconds: number };
  memory: MemoryStatus;
}

// ─── API Client ───────────────────────────────────────────────────────────────

async function fetchHealth(): Promise<SystemHealth> {
  const res = await fetch(`${API_URL}/health`);
  return res.json();
}

async function searchMemory(query: string): Promise<any> {
  const res = await fetch(`${API_URL}/api/memory/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: 5 }),
  });
  return res.json();
}

// ─── Hooks ────────────────────────────────────────────────────────────────────

function useWebSocket(sessionId: string) {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [streamingContent, setStreamingContent] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const onMessageRef = useRef<((msg: Message) => void) | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    const ws = new WebSocket(`${WS_URL}/ws/chat`);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => {
      setConnected(false);
      setTimeout(connect, 3000);
    };
    ws.onerror = () => ws.close();

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "start") {
        setIsStreaming(true);
        setStreamingContent("");
      } else if (data.type === "token") {
        setStreamingContent((prev) => prev + data.content);
      } else if (data.type === "done") {
        setIsStreaming(false);
        if (onMessageRef.current && streamingContent) {
          onMessageRef.current({
            id: data.trace_id,
            role: "assistant",
            content: streamingContent,
            timestamp: new Date().toISOString(),
          });
        }
        setStreamingContent("");
      }
    };
  }, [sessionId]);

  useEffect(() => {
    connect();
    return () => wsRef.current?.close();
  }, [connect]);

  const send = useCallback((content: string, modality: string = "text") => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ session_id: sessionId, content, modality }));
    }
  }, [sessionId]);

  return { connected, send, streamingContent, isStreaming, onMessageRef };
}

// ─── Components ───────────────────────────────────────────────────────────────

function StatusBadge({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 4,
      padding: "2px 8px", borderRadius: 12, fontSize: 12,
      background: ok ? "#d1fae5" : "#fee2e2",
      color: ok ? "#065f46" : "#991b1b",
    }}>
      <span style={{ width: 6, height: 6, borderRadius: "50%", background: ok ? "#10b981" : "#ef4444" }} />
      {label}
    </span>
  );
}

function ChatPanel({ sessionId }: { sessionId: string }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const { connected, send, streamingContent, isStreaming, onMessageRef } = useWebSocket(sessionId);

  onMessageRef.current = (msg) => setMessages((prev) => [...prev, msg]);

  const handleSend = () => {
    if (!input.trim() || !connected) return;
    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: input,
      timestamp: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMsg]);
    send(input);
    setInput("");
  };

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingContent]);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", background: "#fff" }}>
      {/* Header */}
      <div style={{ padding: "12px 16px", borderBottom: "1px solid #e5e7eb", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontWeight: 600, fontSize: 15 }}>Chat</span>
        <StatusBadge ok={connected} label={connected ? "Connected" : "Reconnecting"} />
      </div>

      {/* Messages */}
      <div style={{ flex: 1, overflowY: "auto", padding: 16, display: "flex", flexDirection: "column", gap: 12 }}>
        {messages.length === 0 && (
          <div style={{ color: "#9ca3af", textAlign: "center", marginTop: 40, fontSize: 14 }}>
            Start a conversation with the Living AI System.
          </div>
        )}
        {messages.map((msg) => (
          <div key={msg.id} style={{
            alignSelf: msg.role === "user" ? "flex-end" : "flex-start",
            maxWidth: "80%",
          }}>
            <div style={{
              padding: "10px 14px",
              borderRadius: msg.role === "user" ? "16px 16px 4px 16px" : "16px 16px 16px 4px",
              background: msg.role === "user" ? "#3b82f6" : "#f3f4f6",
              color: msg.role === "user" ? "#fff" : "#111827",
              fontSize: 14,
              lineHeight: 1.6,
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
            }}>
              {msg.content}
            </div>
            {msg.modulesActivated && msg.modulesActivated.length > 0 && (
              <div style={{ marginTop: 4, fontSize: 11, color: "#9ca3af" }}>
                Modules: {msg.modulesActivated.join(", ")}
              </div>
            )}
          </div>
        ))}

        {/* Streaming indicator */}
        {isStreaming && (
          <div style={{ alignSelf: "flex-start", maxWidth: "80%" }}>
            <div style={{
              padding: "10px 14px", borderRadius: "16px 16px 16px 4px",
              background: "#f3f4f6", color: "#111827", fontSize: 14,
              lineHeight: 1.6, whiteSpace: "pre-wrap",
            }}>
              {streamingContent || <span style={{ opacity: 0.5 }}>Thinking...</span>}
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div style={{ padding: 16, borderTop: "1px solid #e5e7eb", display: "flex", gap: 8 }}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
          placeholder="Type a message... (Enter to send, Shift+Enter for new line)"
          style={{
            flex: 1, resize: "none", border: "1px solid #d1d5db",
            borderRadius: 8, padding: "8px 12px", fontSize: 14,
            fontFamily: "inherit", lineHeight: 1.5, minHeight: 44, maxHeight: 200,
          }}
          rows={1}
          disabled={!connected || isStreaming}
        />
        <button
          onClick={handleSend}
          disabled={!connected || !input.trim() || isStreaming}
          style={{
            padding: "0 20px", borderRadius: 8, border: "none",
            background: "#3b82f6", color: "#fff", fontWeight: 600,
            cursor: "pointer", fontSize: 14, height: 44, alignSelf: "flex-end",
            opacity: (!connected || !input.trim() || isStreaming) ? 0.5 : 1,
          }}
        >
          Send
        </button>
      </div>
    </div>
  );
}

function SystemStatusPanel() {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      try {
        const h = await fetchHealth();
        setHealth(h);
      } catch {
        setHealth(null);
      } finally {
        setLoading(false);
      }
    };
    load();
    const interval = setInterval(load, 10000);
    return () => clearInterval(interval);
  }, []);

  if (loading) return <div style={{ padding: 16, color: "#9ca3af" }}>Loading system status...</div>;
  if (!health) return <div style={{ padding: 16, color: "#ef4444" }}>System unreachable</div>;

  const rows = [
    { label: "System Status", value: <StatusBadge ok={health.status === "healthy"} label={health.status} /> },
    { label: "CEE Loop", value: <StatusBadge ok={health.cee?.running} label={health.cee?.running ? "Running" : "Stopped"} /> },
    { label: "CEE Time Steps", value: health.cee?.time_step?.toLocaleString() ?? "0" },
    { label: "Uptime", value: `${health.cee?.uptime_seconds ?? 0}s` },
    { label: "Active Sessions", value: health.memory?.working_memory?.active_sessions ?? 0 },
    { label: "Working Memory Tokens", value: health.memory?.working_memory?.total_tokens?.toLocaleString() ?? 0 },
    { label: "Episodic Memory", value: <StatusBadge ok={health.memory?.episodic_memory?.initialised} label={`${health.memory?.episodic_memory?.entry_count ?? 0} entries`} /> },
    { label: "Knowledge Base", value: <StatusBadge ok={health.memory?.knowledge_base?.initialised} label={`${health.memory?.knowledge_base?.fact_count ?? 0} facts`} /> },
    { label: "Knowledge Graph", value: `${health.memory?.knowledge_base?.graph_nodes ?? 0} nodes` },
  ];

  return (
    <div style={{ padding: 16 }}>
      <div style={{ fontWeight: 600, marginBottom: 12, fontSize: 15 }}>System Status</div>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {rows.map(({ label, value }) => (
          <div key={label} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 13 }}>
            <span style={{ color: "#6b7280" }}>{label}</span>
            <span style={{ fontWeight: 500 }}>{value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function MemoryBrowserPanel() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const search = async () => {
    if (!query.trim()) return;
    setLoading(true);
    try {
      const r = await searchMemory(query);
      setResults(r);
    } catch {
      setResults({ error: "Search failed" });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 16 }}>
      <div style={{ fontWeight: 600, marginBottom: 12, fontSize: 15 }}>Memory Browser</div>
      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") search(); }}
          placeholder="Search memory and knowledge..."
          style={{ flex: 1, border: "1px solid #d1d5db", borderRadius: 6, padding: "6px 10px", fontSize: 13 }}
        />
        <button
          onClick={search}
          disabled={loading}
          style={{ padding: "6px 14px", borderRadius: 6, border: "none", background: "#3b82f6", color: "#fff", cursor: "pointer", fontSize: 13 }}
        >
          {loading ? "..." : "Search"}
        </button>
      </div>

      {results && !results.error && (
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {results.episodic_results?.length > 0 && (
            <div>
              <div style={{ fontSize: 12, fontWeight: 600, color: "#6b7280", marginBottom: 6 }}>EPISODIC MEMORY</div>
              {results.episodic_results.map((r: any, i: number) => (
                <div key={i} style={{ fontSize: 12, padding: "6px 10px", background: "#f9fafb", borderRadius: 6, marginBottom: 4, color: "#374151" }}>
                  {r.content?.slice(0, 200)}
                  {r.similarity && <span style={{ color: "#9ca3af", marginLeft: 8 }}>({(r.similarity * 100).toFixed(0)}% match)</span>}
                </div>
              ))}
            </div>
          )}
          {results.knowledge_results?.length > 0 && (
            <div>
              <div style={{ fontSize: 12, fontWeight: 600, color: "#6b7280", marginBottom: 6 }}>KNOWLEDGE BASE</div>
              {results.knowledge_results.map((r: any, i: number) => (
                <div key={i} style={{ fontSize: 12, padding: "6px 10px", background: "#f0fdf4", borderRadius: 6, marginBottom: 4, color: "#374151" }}>
                  {r.content?.slice(0, 200)}
                </div>
              ))}
            </div>
          )}
          {results.episodic_results?.length === 0 && results.knowledge_results?.length === 0 && (
            <div style={{ color: "#9ca3af", fontSize: 13 }}>No results found.</div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────

export default function App() {
  const [sessionId] = useState(() => crypto.randomUUID());
  const [activeTab, setActiveTab] = useState<"chat" | "memory" | "status">("chat");

  const tabs = [
    { id: "chat", label: "Chat" },
    { id: "memory", label: "Memory" },
    { id: "status", label: "Status" },
  ] as const;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", fontFamily: "system-ui, sans-serif", background: "#f9fafb" }}>
      {/* Top bar */}
      <div style={{ padding: "0 20px", background: "#1e293b", display: "flex", alignItems: "center", gap: 16, height: 52, flexShrink: 0 }}>
        <span style={{ color: "#fff", fontWeight: 700, fontSize: 16 }}>Living AI System</span>
        <span style={{ color: "#64748b", fontSize: 12 }}>The Game of Infinite Paths</span>
        <div style={{ marginLeft: "auto", display: "flex", gap: 4 }}>
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                padding: "4px 14px", borderRadius: 6, border: "none", cursor: "pointer",
                fontSize: 13, fontWeight: activeTab === tab.id ? 600 : 400,
                background: activeTab === tab.id ? "#3b82f6" : "transparent",
                color: activeTab === tab.id ? "#fff" : "#94a3b8",
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflow: "hidden" }}>
        {activeTab === "chat" && (
          <div style={{ height: "100%", maxWidth: 900, margin: "0 auto" }}>
            <ChatPanel sessionId={sessionId} />
          </div>
        )}
        {activeTab === "memory" && (
          <div style={{ height: "100%", maxWidth: 800, margin: "0 auto", overflowY: "auto", background: "#fff" }}>
            <MemoryBrowserPanel />
          </div>
        )}
        {activeTab === "status" && (
          <div style={{ height: "100%", maxWidth: 600, margin: "0 auto", overflowY: "auto", background: "#fff" }}>
            <SystemStatusPanel />
          </div>
        )}
      </div>
    </div>
  );
}
