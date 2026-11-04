"use client";

import { FormEvent, useEffect, useState } from "react";

type User = {
  id: string;
  email: string;
  display_name: string;
  role: "lawyer" | "student";
};

type Citation = {
  id: string;
  title: string;
  section?: string | null;
  paragraph?: string | null;
  pages?: { start?: number | null; end?: number | null };
  source_url?: string | null;
  excerpt?: string | null;
};

type Message = {
  role: "assistant" | "user";
  content: string;
  status?: string | null;
  citations?: Citation[];
  note?: string;
  pending?: boolean;
};

type Conversation = { id: string; title: string; updated_at: string; document_id?: string | null; document_title?: string | null };
type WorkType = "case_brief" | "judgment_analysis" | "note" | "draft" | "other";
type SavedWork = { id: string; work_type: WorkType; title: string; content: string; updated_at: string; citations?: Citation[] };
type JudgmentDocument = { id: string; title: string; page_count: number; chunk_count: number; created_at: string };
type CorpusSource = { title: string; document_type: string; jurisdiction: string; issuing_authority?: string | null; language_code?: string | null; version_label?: string | null; published_on?: string | null; effective_from?: string | null; effective_until?: string | null; source_url?: string | null; chunk_count: number; page_start?: number | null; page_end?: number | null };
type PassageResult = { id: string; title: string; section?: string | null; paragraph?: string | null; pages?: { start?: number | null; end?: number | null }; text: string };
type PassageSearchResponse = { query: string; results: PassageResult[]; message: string };
type BriefResult = { status: string; brief: string | null; citations: Citation[]; saved_work_id: string | null; message: string };
type ChatResult = {
  status: string;
  answer: string | null;
  citations: Citation[];
  message: string;
  conversation_id: string;
};
type LoginResult = { access_token: string; user: User };

const API_BASE_URL = process.env.NEXT_PUBLIC_LAWTEFI_API_URL ?? "http://127.0.0.1:8000";
const starterPrompts = [
  "Find a provision in the BNS",
  "Help me brief a judgment",
  "Explain a constitutional concept",
];

async function apiRequest<T>(path: string, token?: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body) headers.set("Content-Type", "application/json");
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const response = await fetch(API_BASE_URL + path, { ...init, headers });
  const result = (await response.json().catch(() => ({}))) as T & { detail?: string };
  if (!response.ok) throw new Error(result.detail ?? "The request could not be completed.");
  return result;
}

function citationHref(citation: Citation): string {
  if (!citation.source_url) return "#";
  const page = citation.pages?.start;
  try {
    const url = new URL(citation.source_url);
    if (page && /\.pdf(?:$|[?#])/i.test(url.pathname)) url.hash = `page=${page}`;
    return url.toString();
  } catch {
    return citation.source_url;
  }
}

function SourceCitationCard({ citation }: { citation: Citation }) {
  const pages = citation.pages;
  const pageLabel = pages?.start
    ? pages.start === pages.end ? `p. ${pages.start}` : `pp. ${pages.start}–${pages.end}`
    : null;
  return <details className="source-citation">
    <summary className="citation-card">
      <span className="citation-id">{citation.id}</span>
      <span className="citation-details"><strong>{citation.title}</strong><span>{[citation.section, citation.paragraph ? `¶ ${citation.paragraph}` : null, pageLabel].filter(Boolean).join(" · ")}</span></span>
      <span className="citation-arrow" aria-hidden="true">⌄</span>
    </summary>
    <div className="citation-excerpt">
      <p className="eyebrow">Retrieved passage for this citation</p>
      <p>{citation.excerpt || "This saved citation does not include an extracted passage."}</p>
      {citation.source_url && <a href={citationHref(citation)} rel="noreferrer" target="_blank">Open original document ↗</a>}
    </div>
  </details>;
}

function exportSavedWorkMarkdown(title: string, content: string, citations: Citation[]) {
  const sourceLines = citations.flatMap((citation) => {
    const pages = citation.pages;
    const location = [citation.section, citation.paragraph ? `paragraph ${citation.paragraph}` : null,
      pages?.start ? pages.start === pages.end ? `page ${pages.start}` : `pages ${pages.start}–${pages.end}` : null]
      .filter(Boolean).join(" · ");
    const lines = [`### [${citation.id}] ${citation.title}${location ? ` — ${location}` : ""}`];
    if (citation.excerpt) {
      lines.push("", ...citation.excerpt.split("\n").map((line) => `> ${line}`));
    }
    if (citation.source_url) lines.push("", `[Open original source](${citationHref(citation)})`);
    return [...lines, ""];
  });
  const markdown = [
    `# ${title.trim()}`,
    "",
    content.trim(),
    ...(sourceLines.length ? ["", "## Sources", "", ...sourceLines] : []),
  ].join("\n").trim() + "\n";
  const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${title.trim().replace(/[<>:\"/\\|?*\u0000-\u001f]/g, "-").replace(/\s+/g, "-").slice(0, 100) || "lawtefi-research"}.md`;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function uploadPdf(file: File, token: string): Promise<JudgmentDocument> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(API_BASE_URL + "/api/v1/documents/upload", {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
    body: form,
  });
  const result = (await response.json().catch(() => ({}))) as JudgmentDocument & { detail?: string };
  if (!response.ok) throw new Error(result.detail ?? "The judgment could not be uploaded.");
  return result;
}

export default function Home() {
  const [user, setUser] = useState<User | null>(null);
  const [authLoading, setAuthLoading] = useState(true);
  const [authMode, setAuthMode] = useState<"login" | "register">("login");
  const [authName, setAuthName] = useState("");
  const [authEmail, setAuthEmail] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authRole, setAuthRole] = useState<User["role"]>("student");
  const [authError, setAuthError] = useState("");
  const [isAuthenticating, setIsAuthenticating] = useState(false);
  const [token, setToken] = useState("");
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [prompt, setPrompt] = useState("");
  const [notice, setNotice] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [workspaceView, setWorkspaceView] = useState<"assistant" | "saved-work" | "documents" | "sources">("assistant");
  const [savedWork, setSavedWork] = useState<SavedWork[]>([]);
  const [selectedWorkId, setSelectedWorkId] = useState<string | null>(null);
  const [workType, setWorkType] = useState<WorkType>("note");
  const [workTitle, setWorkTitle] = useState("");
  const [workContent, setWorkContent] = useState("");
  const [workNotice, setWorkNotice] = useState("");
  const [workSaving, setWorkSaving] = useState(false);
  const [workCitations, setWorkCitations] = useState<Citation[]>([]);
  const [savingAnswerKey, setSavingAnswerKey] = useState<string | null>(null);
  const [savedAnswerKeys, setSavedAnswerKeys] = useState<Set<string>>(new Set());
  const [documents, setDocuments] = useState<JudgmentDocument[]>([]);
  const [corpusSources, setCorpusSources] = useState<CorpusSource[]>([]);
  const [sourceNotice, setSourceNotice] = useState("");
  const [activeDocumentId, setActiveDocumentId] = useState<string | null>(null);
  const [activeDocumentTitle, setActiveDocumentTitle] = useState("");
  const [selectedUpload, setSelectedUpload] = useState<File | null>(null);
  const [documentBusy, setDocumentBusy] = useState(false);
  const [documentNotice, setDocumentNotice] = useState("");
  const [briefResult, setBriefResult] = useState<BriefResult | null>(null);
  const [passageSearchDocumentId, setPassageSearchDocumentId] = useState<string | null>(null);
  const [passageSearchQuery, setPassageSearchQuery] = useState("");
  const [passageSearchResults, setPassageSearchResults] = useState<PassageResult[]>([]);
  const [passageSearchNotice, setPassageSearchNotice] = useState("");
  const [passageSearchBusy, setPassageSearchBusy] = useState(false);

  async function loadConversationList(accessToken: string) {
    const rows = await apiRequest<Conversation[]>("/api/v1/conversations", accessToken);
    setConversations(rows);
    return rows;
  }

  async function loadConversation(conversation: Conversation, accessToken: string) {
    const rows = await apiRequest<Array<Message & { citations?: Citation[] }>>(
      `/api/v1/conversations/${conversation.id}/messages`,
      accessToken,
    );
    setConversationId(conversation.id);
    setActiveDocumentId(conversation.document_id ?? null);
    setActiveDocumentTitle(conversation.document_title ?? "");
    setMessages(rows.map((message) => ({
      role: message.role,
      content: message.content,
      status: message.status,
      citations: message.citations ?? [],
    })));
    setNotice("");
  }

  async function openSavedWork() {
    setWorkspaceView("saved-work");
    setWorkNotice("");
    try {
      const rows = await apiRequest<SavedWork[]>("/api/v1/saved-work", token);
      setSavedWork(rows);
      if (rows.length) {
        const selected = rows.find((item) => item.id === selectedWorkId) ?? rows[0];
        setSelectedWorkId(selected.id);
        setWorkType(selected.work_type);
        setWorkTitle(selected.title);
        setWorkContent(selected.content);
        setWorkCitations(selected.citations ?? []);
      } else {
        setSelectedWorkId(null);
        setWorkType("note");
        setWorkTitle("");
        setWorkContent("");
        setWorkCitations([]);
      }
    } catch (error) {
      setWorkNotice(error instanceof Error ? error.message : "Could not load saved work.");
    }
  }

  async function openCorpusSources() {
    setWorkspaceView("sources");
    setSourceNotice("");
    try {
      const rows = await apiRequest<CorpusSource[]>("/api/v1/sources", token);
      setCorpusSources(rows);
    } catch (error) {
      setSourceNotice(error instanceof Error ? error.message : "Could not load shared legal sources.");
    }
  }

  function selectSavedWork(item: SavedWork) {
    setSelectedWorkId(item.id);
    setWorkType(item.work_type);
    setWorkTitle(item.title);
    setWorkContent(item.content);
    setWorkCitations(item.citations ?? []);
    setWorkNotice("");
  }

  function newSavedWork() {
    setSelectedWorkId(null);
    setWorkType("note");
    setWorkTitle("");
    setWorkContent("");
    setWorkCitations([]);
    setWorkNotice("");
  }

  async function saveWork(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!workTitle.trim()) return;
    setWorkSaving(true);
    setWorkNotice("");
    try {
      const item = selectedWorkId
        ? await apiRequest<SavedWork>(`/api/v1/saved-work/${selectedWorkId}`, token, {
            method: "PATCH",
            body: JSON.stringify({ title: workTitle, content: workContent }),
          })
        : await apiRequest<SavedWork>("/api/v1/saved-work", token, {
            method: "POST",
            body: JSON.stringify({ work_type: workType, title: workTitle, content: workContent }),
          });
      setSelectedWorkId(item.id);
      setWorkType(item.work_type);
      setWorkTitle(item.title);
      setWorkContent(item.content);
      setWorkCitations(item.citations ?? []);
      setSavedWork((current) => [item, ...current.filter((entry) => entry.id !== item.id)]);
      setWorkNotice("Saved to your private workspace.");
    } catch (error) {
      setWorkNotice(error instanceof Error ? error.message : "Could not save this item.");
    } finally {
      setWorkSaving(false);
    }
  }

  async function saveGroundedAnswer(message: Message, question: string, key: string) {
    if (!message.citations?.length || savedAnswerKeys.has(key)) return;
    setSavingAnswerKey(key);
    setNotice("");
    try {
      const item = await apiRequest<SavedWork>("/api/v1/saved-work", token, {
        method: "POST",
        body: JSON.stringify({
          work_type: activeDocumentId ? "judgment_analysis" : "note",
          title: `Research: ${question}`.slice(0, 200),
          content: message.content,
          citations: message.citations,
        }),
      });
      setSavedWork((current) => [item, ...current.filter((entry) => entry.id !== item.id)]);
      setSavedAnswerKeys((current) => new Set(current).add(key));
      setNotice("Saved this source-linked answer to My saved work.");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not save this answer.");
    } finally {
      setSavingAnswerKey(null);
    }
  }

  async function removeSavedWork() {
    if (!selectedWorkId || !window.confirm("Delete this saved item? This cannot be undone.")) return;
    try {
      await apiRequest<unknown>(`/api/v1/saved-work/${selectedWorkId}`, token, { method: "DELETE" });
      const remaining = savedWork.filter((item) => item.id !== selectedWorkId);
      setSavedWork(remaining);
      if (remaining.length) selectSavedWork(remaining[0]);
      else newSavedWork();
      setWorkNotice("Saved item deleted.");
    } catch (error) {
      setWorkNotice(error instanceof Error ? error.message : "Could not delete this item.");
    }
  }

  async function openDocuments() {
    setWorkspaceView("documents");
    setDocumentNotice("");
    setBriefResult(null);
    try {
      setDocuments(await apiRequest<JudgmentDocument[]>("/api/v1/documents", token));
    } catch (error) {
      setDocumentNotice(error instanceof Error ? error.message : "Could not load your judgment documents.");
    }
  }

  async function uploadJudgment(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedUpload) return;
    setDocumentBusy(true);
    setDocumentNotice("Extracting pages and building a private local index…");
    setBriefResult(null);
    try {
      const uploadedDocument = await uploadPdf(selectedUpload, token);
      setDocuments((current) => [uploadedDocument, ...current.filter((item) => item.id !== uploadedDocument.id)]);
      setSelectedUpload(null);
      const input = document.getElementById("judgment-pdf");
      if (input instanceof HTMLInputElement) input.value = "";
      setDocumentNotice(`Indexed ${uploadedDocument.page_count} pages into ${uploadedDocument.chunk_count} passages. This document is private to your account.`);
    } catch (error) {
      setDocumentNotice(error instanceof Error ? error.message : "Could not upload this judgment.");
    } finally {
      setDocumentBusy(false);
    }
  }

  function askAboutJudgment(document: JudgmentDocument) {
    setActiveDocumentId(document.id);
    setActiveDocumentTitle(document.title);
    setConversationId(null);
    setMessages([]);
    setPrompt("");
    setNotice(`Questions in this chat will use only ${document.title}.`);
    setWorkspaceView("assistant");
  }

  async function searchJudgmentPassages(event: FormEvent<HTMLFormElement>, documentId: string) {
    event.preventDefault();
    const query = passageSearchQuery.trim();
    if (!query || !token) return;
    setPassageSearchBusy(true);
    setPassageSearchNotice("Searching the private local index…");
    setPassageSearchResults([]);
    try {
      const result = await apiRequest<PassageSearchResponse>(`/api/v1/documents/${documentId}/search`, token, {
        method: "POST",
        body: JSON.stringify({ query }),
      });
      setPassageSearchResults(result.results);
      setPassageSearchNotice(result.results.length ? result.message : "No relevant indexed passages were found. Try different wording.");
    } catch (error) {
      setPassageSearchNotice(error instanceof Error ? error.message : "Could not search this judgment.");
    } finally {
      setPassageSearchBusy(false);
    }
  }

  async function generateBrief(document: JudgmentDocument) {
    setDocumentBusy(true);
    setDocumentNotice("Preparing a source-linked case brief locally…");
    setBriefResult(null);
    try {
      const result = await apiRequest<BriefResult>(`/api/v1/documents/${document.id}/brief`, token, {
        method: "POST",
        body: JSON.stringify({}),
      });
      setBriefResult(result);
      setDocumentNotice(result.message);
    } catch (error) {
      setDocumentNotice(error instanceof Error ? error.message : "Could not create a case brief from this judgment.");
    } finally {
      setDocumentBusy(false);
    }
  }

  async function removeDocument(document: JudgmentDocument) {
    if (!window.confirm(`Delete “${document.title}” and its private index? This cannot be undone.`)) return;
    setDocumentBusy(true);
    setDocumentNotice("");
    try {
      await apiRequest<unknown>(`/api/v1/documents/${document.id}`, token, { method: "DELETE" });
      setDocuments((current) => current.filter((item) => item.id !== document.id));
      if (activeDocumentId === document.id) {
        setActiveDocumentId(null);
        setActiveDocumentTitle("");
      }
      if (passageSearchDocumentId === document.id) {
        setPassageSearchDocumentId(null);
        setPassageSearchQuery("");
        setPassageSearchResults([]);
        setPassageSearchNotice("");
      }
      setDocumentNotice("Judgment and its private index deleted.");
      setBriefResult(null);
    } catch (error) {
      setDocumentNotice(error instanceof Error ? error.message : "Could not delete this judgment.");
    } finally {
      setDocumentBusy(false);
    }
  }

  useEffect(() => {
    const storedToken = window.localStorage.getItem("lawtefi_access_token");
    if (!storedToken) {
      setAuthLoading(false);
      return;
    }
    setToken(storedToken);
    void (async () => {
      try {
        const currentUser = await apiRequest<User>("/api/v1/auth/me", storedToken);
        setUser(currentUser);
        const rows = await loadConversationList(storedToken);
        if (rows.length) await loadConversation(rows[0], storedToken);
      } catch {
        window.localStorage.removeItem("lawtefi_access_token");
        setToken("");
        setUser(null);
      } finally {
        setAuthLoading(false);
      }
    })();
    // Session restoration intentionally runs once on page load.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function submitAuth(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsAuthenticating(true);
    setAuthError("");
    try {
      if (authMode === "register") {
        await apiRequest<User>("/api/v1/auth/register", undefined, {
          method: "POST",
          body: JSON.stringify({
            email: authEmail,
            display_name: authName,
            role: authRole,
            password: authPassword,
          }),
        });
      }
      const login = await apiRequest<LoginResult>("/api/v1/auth/login", undefined, {
        method: "POST",
        body: JSON.stringify({ email: authEmail, password: authPassword }),
      });
      window.localStorage.setItem("lawtefi_access_token", login.access_token);
      setToken(login.access_token);
      setUser(login.user);
      setWorkspaceView("assistant");
      setSavedWork([]);
      setSelectedWorkId(null);
      setWorkTitle("");
      setWorkContent("");
      setWorkCitations([]);
      setDocuments([]);
      setActiveDocumentId(null);
      setActiveDocumentTitle("");
      setSelectedUpload(null);
      setDocumentNotice("");
      setBriefResult(null);
      setPassageSearchDocumentId(null);
      setPassageSearchQuery("");
      setPassageSearchResults([]);
      setPassageSearchNotice("");
      const rows = await loadConversationList(login.access_token);
      if (rows.length) await loadConversation(rows[0], login.access_token);
      else {
        setConversationId(null);
        setMessages([]);
      }
    } catch (error) {
      setAuthError(error instanceof Error ? error.message : "Could not sign in to LawTeFI.");
    } finally {
      setIsAuthenticating(false);
    }
  }

  async function submitQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const question = prompt.trim();
    if (!question || isSending || !token) return;
    setIsSending(true);
    setMessages((current) => [
      ...current,
      { role: "user", content: question },
      { role: "assistant", content: "Searching the local legal collection…", pending: true },
    ]);
    setPrompt("");
    setNotice("");
    try {
      const result = await apiRequest<ChatResult>("/api/v1/chat", token, {
        method: "POST",
        body: JSON.stringify({ question, conversation_id: conversationId, document_id: activeDocumentId }),
      });
      setConversationId(result.conversation_id);
      setMessages((current) => {
        const updated = [...current];
        const lastAssistant = updated.findLastIndex((message) => message.role === "assistant");
        if (lastAssistant >= 0) {
          updated[lastAssistant] = {
            role: "assistant",
            content: result.answer ?? result.message,
            status: result.status,
            citations: result.citations ?? [],
            note: result.answer ? result.message : undefined,
          };
        }
        return updated;
      });
      await loadConversationList(token);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Could not reach the LawTeFI API.";
      setMessages((current) => {
        const updated = [...current];
        const lastAssistant = updated.findLastIndex((message) => message.role === "assistant");
        if (lastAssistant >= 0) {
          updated[lastAssistant] = {
            role: "assistant",
            content: "I couldn't complete the legal research request.",
            note: errorMessage,
          };
        }
        return updated;
      });
    } finally {
      setIsSending(false);
    }
  }

  function startNewChat() {
    setConversationId(null);
    setMessages([]);
    setNotice("");
  }

  function signOut() {
    window.localStorage.removeItem("lawtefi_access_token");
    setToken("");
    setUser(null);
    setConversations([]);
    setConversationId(null);
    setMessages([]);
    setWorkspaceView("assistant");
    setSavedWork([]);
    setSelectedWorkId(null);
    setWorkTitle("");
    setWorkContent("");
    setWorkCitations([]);
    setDocuments([]);
    setActiveDocumentId(null);
    setActiveDocumentTitle("");
    setSelectedUpload(null);
    setDocumentNotice("");
    setBriefResult(null);
    setPassageSearchDocumentId(null);
    setPassageSearchQuery("");
    setPassageSearchResults([]);
    setPassageSearchNotice("");
    setWorkNotice("");
  }

  if (authLoading) return <main className="auth-loading">Opening your LawTeFI workspace…</main>;

  if (!user) {
    return (
      <main className="auth-page">
        <section className="auth-card">
          <a className="brand auth-brand" href="#home" aria-label="LawTeFI home">
            <span className="brand-mark">L</span><span>LawTeFI</span>
          </a>
          <p className="eyebrow auth-eyebrow">Private legal workspace</p>
          <h1>{authMode === "login" ? "Welcome back" : "Create your workspace"}</h1>
          <p className="auth-copy">Sign in to continue your source-grounded legal research.</p>
          <form className="auth-form" onSubmit={submitAuth}>
            {authMode === "register" && (
              <>
                <label>Full name<input value={authName} onChange={(event) => setAuthName(event.target.value)} autoComplete="name" required /></label>
                <label>Your workspace<select value={authRole} onChange={(event) => setAuthRole(event.target.value as User["role"])}>
                  <option value="student">Law student</option><option value="lawyer">Lawyer</option>
                </select></label>
              </>
            )}
            <label>Email address<input type="email" value={authEmail} onChange={(event) => setAuthEmail(event.target.value)} autoComplete="email" required /></label>
            <label>Password<input type="password" value={authPassword} onChange={(event) => setAuthPassword(event.target.value)} autoComplete={authMode === "login" ? "current-password" : "new-password"} minLength={8} required /></label>
            {authError && <p className="auth-error" role="alert">{authError}</p>}
            <button className="auth-submit" type="submit" disabled={isAuthenticating}>
              {isAuthenticating ? "Please wait…" : authMode === "login" ? "Sign in" : "Create account"}
            </button>
          </form>
          <p className="auth-switch">
            {authMode === "login" ? "New to LawTeFI?" : "Already have an account?"}{" "}
            <button type="button" onClick={() => { setAuthError(""); setAuthMode(authMode === "login" ? "register" : "login"); }}>
              {authMode === "login" ? "Create an account" : "Sign in"}
            </button>
          </p>
        </section>
      </main>
    );
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#home" aria-label="LawTeFI home"><span className="brand-mark">L</span><span>LawTeFI</span></a>
        <button className="new-chat" onClick={startNewChat}><span aria-hidden="true">＋</span> New conversation</button>
        <div className="sidebar-section">
          <p className="eyebrow">Workspace</p>
          <button className={`nav-item ${workspaceView === "assistant" ? "active" : ""}`} onClick={() => setWorkspaceView("assistant")}><span>◉</span> Legal assistant</button>
          <button className={`nav-item ${workspaceView === "saved-work" ? "active" : ""}`} onClick={() => void openSavedWork()}><span>▤</span> My saved work</button>
          <button className={`nav-item ${workspaceView === "documents" ? "active" : ""}`} onClick={() => void openDocuments()}><span>▧</span> My judgments</button>
          <button className={`nav-item ${workspaceView === "sources" ? "active" : ""}`} onClick={() => void openCorpusSources()}><span>▥</span> Legal sources</button>
        </div>
        {workspaceView === "assistant" && <div className="sidebar-section history-section">
          <p className="eyebrow">Recent conversations</p>
          {conversations.length ? conversations.map((conversation) => (
            <button
              className={`history-item ${conversation.id === conversationId ? "selected" : ""}`}
              key={conversation.id}
              onClick={() => void loadConversation(conversation, token).catch((error: Error) => setNotice(error.message))}
            >{conversation.title}</button>
          )) : <p className="empty-history">Your conversations will appear here.</p>}
        </div>}
        <div className="profile-card">
          <div className="avatar">{user.display_name.slice(0, 1).toUpperCase()}</div>
          <div><strong>{user.display_name}</strong><span>{user.role === "student" ? "Student workspace" : "Lawyer workspace"}</span></div>
          <button className="profile-menu" onClick={signOut} aria-label="Sign out" title="Sign out">↪</button>
        </div>
      </aside>

      <section className="main-panel" id="home">
        <header className="topbar">
          <div className="workspace-label"><span className="status-dot" /> Private workspace <span className="divider">/</span> {workspaceView === "assistant" ? (activeDocumentTitle || (user.role === "student" ? "Student" : "Lawyer")) : workspaceView === "documents" ? "My judgments" : workspaceView === "sources" ? "Legal sources" : "Saved work"}</div>
          <button className="role-switch" onClick={signOut}>Sign out</button>
        </header>
        {workspaceView === "saved-work" ? (
          <div className="saved-work-area">
            <section className="saved-work-library">
              <div className="saved-work-heading"><div><p className="eyebrow">Your workspace</p><h2>My saved work</h2></div><button className="work-new-button" onClick={newSavedWork}>＋ New</button></div>
              {savedWork.length ? savedWork.map((item) => (
                <button className={`saved-work-item ${item.id === selectedWorkId ? "selected" : ""}`} key={item.id} onClick={() => selectSavedWork(item)}>
                  <span className="work-kind">{item.work_type.replaceAll("_", " ")}</span><strong>{item.title}</strong>
                  <span>{item.content.trim().slice(0, 72) || "No content yet"}</span>
                </button>
              )) : <p className="empty-work">Save notes, drafts, case briefs, and judgment analyses here.</p>}
            </section>
            <section className="saved-work-editor">
              <div className="saved-work-editor-heading"><div><p className="eyebrow">Private to your account</p><h2>{selectedWorkId ? "Edit saved work" : "Create saved work"}</h2></div>
                {selectedWorkId && <button className="work-delete-button" type="button" onClick={() => void removeSavedWork()}>Delete</button>}
              </div>
              <form onSubmit={saveWork}>
                <label className="work-field">Type<select value={workType} disabled={Boolean(selectedWorkId)} onChange={(event) => setWorkType(event.target.value as WorkType)}>
                  <option value="note">Note</option><option value="draft">Draft</option><option value="case_brief">Case brief</option><option value="judgment_analysis">Judgment analysis</option><option value="other">Other</option>
                </select></label>
                <label className="work-field">Title<input value={workTitle} onChange={(event) => setWorkTitle(event.target.value)} maxLength={200} placeholder="Give this item a title" required /></label>
                <label className="work-field">Content<textarea value={workContent} onChange={(event) => setWorkContent(event.target.value)} maxLength={100000} placeholder="Write your notes or draft here…" rows={15} /></label>
                {workNotice && <p className="work-notice" role="status">{workNotice}</p>}
                <button className="work-save-button" type="submit" disabled={workSaving || !workTitle.trim()}>{workSaving ? "Saving…" : "Save to my workspace"}</button>
                <button className="work-export-button" type="button" onClick={() => exportSavedWorkMarkdown(workTitle, workContent, workCitations)} disabled={!workTitle.trim()}>Download as Markdown</button>
              </form>
              {workCitations.length > 0 && <div className="work-source-list"><p className="eyebrow">Brief sources</p>{workCitations.map((citation) => {
                return <SourceCitationCard citation={citation} key={citation.id} />;
              })}</div>}
            </section>
          </div>
        ) : workspaceView === "documents" ? (
          <div className="documents-area">
            <section className="document-library">
              <div className="document-heading"><div><p className="eyebrow">Private to your account</p><h2>Judgment documents</h2></div></div>
              <form className="upload-panel" onSubmit={uploadJudgment}>
                <label htmlFor="judgment-pdf">Upload a text-based PDF judgment</label>
                <input id="judgment-pdf" type="file" accept="application/pdf,.pdf" onChange={(event) => setSelectedUpload(event.target.files?.[0] ?? null)} />
                <p>LawTeFI extracts pages and builds a private local search index. Scanned PDFs are not supported yet.</p>
                <button className="work-save-button" type="submit" disabled={!selectedUpload || documentBusy}>{documentBusy ? "Working locally…" : "Upload and index"}</button>
              </form>
              {documentNotice && <p className="document-notice" role="status">{documentNotice}</p>}
              <div className="document-list">
                {documents.length ? documents.map((item) => (
                  <article className="document-item" key={item.id}>
                    <div className="document-item-copy"><span className="work-kind">{item.page_count} pages · {item.chunk_count} passages</span><strong>{item.title}</strong></div>
                    <div className="document-actions">
                      <button type="button" onClick={() => askAboutJudgment(item)}>Ask this judgment</button>
                      <button type="button" onClick={() => {
                        setPassageSearchDocumentId((current) => current === item.id ? null : item.id);
                        setPassageSearchResults([]);
                        setPassageSearchNotice("");
                      }}>{passageSearchDocumentId === item.id ? "Close passage search" : "Search passages"}</button>
                      <button type="button" onClick={() => void generateBrief(item)} disabled={documentBusy}>Case brief (incomplete)</button>
                      <button type="button" className="document-delete" onClick={() => void removeDocument(item)} disabled={documentBusy}>Delete</button>
                    </div>
                    {passageSearchDocumentId === item.id && <div className="passage-search-panel">
                      <p className="eyebrow">Search within this judgment</p>
                      <form className="passage-search-form" onSubmit={(event) => void searchJudgmentPassages(event, item.id)}>
                        <input
                          value={passageSearchQuery}
                          onChange={(event) => setPassageSearchQuery(event.target.value)}
                          placeholder="Enter a legal phrase or question"
                          minLength={2}
                          maxLength={500}
                          required
                        />
                        <button type="submit" disabled={passageSearchBusy}>{passageSearchBusy ? "Searching…" : "Find passages"}</button>
                      </form>
                      {passageSearchNotice && <p className="passage-search-notice" role="status">{passageSearchNotice}</p>}
                      {passageSearchResults.map((result) => {
                        const pageLabel = result.pages?.start ? result.pages.start === result.pages.end ? `p. ${result.pages.start}` : `pp. ${result.pages.start}–${result.pages.end}` : "Page unavailable";
                        return <article className="passage-result" key={result.id}>
                          <div className="passage-result-heading"><span>{result.id}</span><strong>{result.title}</strong><small>{[result.section, result.paragraph ? `¶ ${result.paragraph}` : null, pageLabel].filter(Boolean).join(" · ")}</small></div>
                          <p>{result.text}</p>
                        </article>;
                      })}
                    </div>}
                  </article>
                )) : <p className="empty-work">Uploaded judgments will be listed here.</p>}
              </div>
            </section>
            {briefResult?.brief && <section className="brief-preview">
              <p className="eyebrow">{briefResult.saved_work_id ? "Saved to My saved work" : "Partial draft · review before saving"}</p><h2>Generated case brief</h2>
              <p className="message-content">{briefResult.brief}</p>
              <div className="citation-list" aria-label="Brief sources">{briefResult.citations.map((citation) => {
                return <SourceCitationCard citation={citation} key={citation.id} />;
              })}</div>
            </section>}
          </div>
        ) : workspaceView === "sources" ? (
          <div className="sources-area">
            <section className="source-library">
              <div className="source-library-heading"><div><p className="eyebrow">Shared and locally indexed</p><h2>Legal sources</h2><p>These are the public documents currently available to LawTeFI’s legal assistant. Personal judgment uploads stay in your private workspace.</p></div></div>
              {sourceNotice && <p className="document-notice" role="status">{sourceNotice}</p>}
              <div className="source-catalog">
                {corpusSources.length ? corpusSources.map((source) => (
                  <article className="source-catalog-card" key={`${source.title}-${source.version_label ?? "current"}`}>
                    <div className="source-catalog-top"><span className="work-kind">{source.document_type}</span><span>{source.jurisdiction}</span></div>
                    <h3>{source.title}</h3>
                    {source.issuing_authority && <p className="source-authority">{source.issuing_authority}</p>}
                    {source.version_label && <p className="source-version">{source.version_label}</p>}
                    <div className="source-catalog-stats"><span>{source.chunk_count} passages</span><span>{source.page_start && source.page_end ? `Indexed pages ${source.page_start}–${source.page_end}` : "Page metadata unavailable"}</span><span>{source.language_code ?? "Language not recorded"}</span></div>
                    {(source.effective_from || source.effective_until || source.published_on) && <p className="source-dates">{[
                      source.published_on ? `Published ${source.published_on}` : null,
                      source.effective_from ? `Effective from ${source.effective_from}` : null,
                      source.effective_until ? `Effective until ${source.effective_until}` : null,
                    ].filter(Boolean).join(" · ")}</p>}
                  </article>
                )) : !sourceNotice && <p className="empty-work">No public legal sources have been indexed yet.</p>}
              </div>
            </section>
          </div>
        ) : <div className="content-area">
          <div className="conversation">
            {messages.length === 0 ? (
              <div className="welcome">
                <div className="welcome-icon">§</div>
                <p className="eyebrow">Legal research, grounded in sources</p>
                <h1>What are you working on?</h1>
                <p className="welcome-copy">{activeDocumentTitle ? `Ask questions grounded only in ${activeDocumentTitle}.` : "Ask a question about the current legal collection. LawTeFI searches its local sources and displays the passages used for each response. This workspace currently includes the Bharatiya Nyaya Sanhita."}</p>
                <div className="prompt-grid">{starterPrompts.map((item) => (
                  <button key={item} className="prompt-card" onClick={() => setPrompt(item)}><span>{item}</span><span className="arrow">↗</span></button>
                ))}</div>
              </div>
            ) : (
              <div className="message-list">{messages.map((message, index) => {
                const previousQuestion = index > 0 && messages[index - 1].role === "user" ? messages[index - 1].content : "";
                const saveKey = `${conversationId ?? "new"}:${index}`;
                return (
                <article className={`message ${message.role}`} key={`${message.role}-${index}`}>
                  <div className="message-avatar">{message.role === "user" ? user.display_name.slice(0, 1).toUpperCase() : "§"}</div>
                  <div className="message-body">
                    <p className="message-label">{message.role === "user" ? "You" : message.pending ? "LawTeFI · Searching" : message.status === "evidence_only" ? "LawTeFI · Statutory source text" : message.citations?.length ? "LawTeFI · Source-grounded draft" : "LawTeFI"}</p>
                    <p className="message-content">{message.content}</p>
                    {message.note && <p className="message-note">{message.note}</p>}
                    {message.role === "assistant" && !message.pending && message.citations?.length && previousQuestion && (
                      <button
                        className="save-answer-button"
                        type="button"
                        disabled={savingAnswerKey === saveKey || savedAnswerKeys.has(saveKey)}
                        onClick={() => void saveGroundedAnswer(message, previousQuestion, saveKey)}
                      >
                        {savingAnswerKey === saveKey ? "Saving…" : savedAnswerKeys.has(saveKey) ? "Saved to My saved work" : "Save to My saved work"}
                      </button>
                    )}
                    {message.citations && message.citations.length > 0 && (
                      <div className="citation-list" aria-label="Sources">{message.citations.map((citation) => {
                        return <SourceCitationCard citation={citation} key={citation.id} />;
                      })}</div>
                    )}
                  </div>
                </article>
                );
              })}</div>
            )}
          </div>
          <div className="composer-wrap">
            {notice && <p className="preview-notice">ⓘ {notice}</p>}
            <form className="composer" onSubmit={submitQuestion}>
              <textarea aria-label="Ask a legal research question" placeholder="Ask a legal research question..." rows={2} value={prompt} onChange={(event) => setPrompt(event.target.value)} disabled={isSending} onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); event.currentTarget.form?.requestSubmit(); }
              }} />
              <div className="composer-footer"><span className="source-hint"><span className="source-lock">▣</span> Retrieved legal passages appear under each response</span>
                <button className="send-button" type="submit" aria-label="Send question" disabled={!prompt.trim() || isSending}>{isSending ? "…" : "↑"}</button>
              </div>
            </form>
            <p className="disclaimer">LawTeFI is a research aid. Always verify legal sources and consult a qualified professional.</p>
          </div>
        </div>}
      </section>
    </main>
  );
}
