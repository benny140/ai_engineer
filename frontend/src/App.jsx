import { useMemo, useRef, useState } from 'react'
import {
  Alert,
  AppBar,
  Avatar,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Container,
  IconButton,
  Paper,
  Stack,
  Switch,
  TextField,
  Toolbar,
  Tooltip,
  Typography,
} from '@mui/material'
import SendRoundedIcon from '@mui/icons-material/SendRounded'
import RestartAltRoundedIcon from '@mui/icons-material/RestartAltRounded'
import SecurityRoundedIcon from '@mui/icons-material/SecurityRounded'
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined'
import './App.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

const STARTER_PROMPTS = [
  'Explain the 6 NIST CSF 2.0 core functions in plain language.',
  'How should a small business prioritize Identify and Protect outcomes?',
  'Give me a sample profile for a cloud-first SaaS startup.',
  'What metrics should we track for Detect and Respond maturity?',
]

function formatTraceDetails(traceItem) {
  const details = { ...traceItem }
  delete details.seq
  delete details.timestamp
  delete details.stage
  delete details.event
  return JSON.stringify(details, null, 2)
}

function summarizeTrace(traceItem) {
  const parts = []
  if (typeof traceItem.duration_ms === 'number') {
    parts.push(`${traceItem.duration_ms} ms`)
  } else if (typeof traceItem.elapsed_ms === 'number') {
    parts.push(`${traceItem.elapsed_ms} ms`)
  }
  if (typeof traceItem.chunk_count === 'number') {
    parts.push(`chunks=${traceItem.chunk_count}`)
  }
  if (typeof traceItem.attempt === 'number' && typeof traceItem.max_attempts === 'number') {
    parts.push(`attempt ${traceItem.attempt}/${traceItem.max_attempts}`)
  }
  return parts.join(' · ')
}

const STAGE_META = {
  planner: { color: '#2563eb', label: 'Planner' },
  retriever: { color: '#0d9488', label: 'Retriever' },
  responder: { color: '#7c3aed', label: 'Responder' },
  pipeline: { color: '#16a34a', label: 'Pipeline' },
}

function getStageMeta(stage) {
  return STAGE_META[stage] || { color: '#64748b', label: stage || 'Event' }
}

function isErrorEvent(event) {
  return typeof event === 'string' && /error|retry|fail|fallback/i.test(event)
}

function TraceTimeline({ items, idPrefix, defaultOpen = false }) {
  return (
    <Box className="trace-timeline">
      {items.map((item, index) => {
        const summary = summarizeTrace(item)
        const meta = getStageMeta(item.stage)
        const errored = isErrorEvent(item.event)
        const accent = errored ? '#dc2626' : meta.color
        return (
          <Box className="trace-node" key={`${idPrefix}-${index}-${item.seq ?? index}`}>
            <span className="trace-dot" style={{ backgroundColor: accent }} />
            <details
              className="trace-details"
              style={{ borderLeftColor: accent }}
              open={defaultOpen}
            >
              <summary className="trace-summary">
                <span className="trace-stage-badge" style={{ backgroundColor: accent }}>
                  {meta.label}
                </span>
                <span className="trace-event-name">{item.event || 'event'}</span>
                {summary ? <span className="trace-summary-meta">{summary}</span> : null}
              </summary>
              <Box component="pre" className="trace-pre">
                {formatTraceDetails(item)}
              </Box>
            </details>
          </Box>
        )
      })}
    </Box>
  )
}

function createSessionId() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID()
  }
  return `session-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
}

function starterMessage() {
  return {
    id: 'assistant-welcome',
    role: 'assistant',
    text: 'Ask about NIST Cybersecurity Framework 2.0. I can help with profiles, controls mapping, implementation steps, and maturity planning.',
    sources: [],
    trace: [],
  }
}

function App() {
  const [messages, setMessages] = useState([starterMessage()])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const [liveTrace, setLiveTrace] = useState([])
  const [includeHistory, setIncludeHistory] = useState(true)
  const [sessionId, setSessionId] = useState(() => createSessionId())

  const listRef = useRef(null)

  const historyLabel = useMemo(
    () =>
      includeHistory
        ? 'Include history is ON: earlier messages in this chat are remembered and can influence future answers.'
        : 'Include history is OFF: each new message is treated independently and previous chat will not influence answers.',
    [includeHistory],
  )

  const historyTooltipText =
    'Include history means the backend keeps this current chat context and uses it to shape future responses. Turn it off for one-shot, stateless questions.'

  const scrollToBottom = () => {
    const el = listRef.current
    if (!el) {
      return
    }
    el.scrollTop = el.scrollHeight
  }

  const appendMessage = (message) => {
    setMessages((prev) => {
      const next = [...prev, message]
      setTimeout(scrollToBottom, 0)
      return next
    })
  }

  const handleReset = () => {
    setMessages([starterMessage()])
    setInput('')
    setError('')
    setSessionId(createSessionId())
  }

  const handleHistoryToggle = (event) => {
    const next = event.target.checked
    setIncludeHistory(next)
    if (next) {
      setSessionId(createSessionId())
    }
  }

  const sendQuestion = async (questionText) => {
    const question = questionText.trim()
    if (!question || isLoading) {
      return
    }

    setError('')
    appendMessage({
      id: `user-${Date.now()}`,
      role: 'user',
      text: question,
      sources: [],
      trace: [],
    })
    setInput('')
    setIsLoading(true)
    setLiveTrace([])

    try {
      const payload = { question }
      if (includeHistory) {
        payload.session_id = sessionId
      }

      const response = await fetch(`${API_BASE_URL}/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })

      if (!response.ok) {
        const detail = await response.text()
        throw new Error(detail || 'Request failed')
      }

      if (!response.body) {
        throw new Error('Streaming response body is not available.')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let completed = false
      const traces = []

      while (true) {
        const { done, value } = await reader.read()
        if (done) {
          break
        }

        buffer += decoder.decode(value, { stream: true })

        while (buffer.includes('\n\n')) {
          const separatorIndex = buffer.indexOf('\n\n')
          const block = buffer.slice(0, separatorIndex)
          buffer = buffer.slice(separatorIndex + 2)

          if (!block.trim()) {
            continue
          }

          const lines = block.split('\n')
          let eventName = 'message'
          const dataLines = []

          for (const line of lines) {
            if (line.startsWith('event:')) {
              eventName = line.slice(6).trim()
            } else if (line.startsWith('data:')) {
              dataLines.push(line.slice(5).trim())
            }
          }

          let payloadData = {}
          if (dataLines.length) {
            try {
              payloadData = JSON.parse(dataLines.join('\n'))
            } catch {
              payloadData = {}
            }
          }

          if (eventName === 'trace') {
            traces.push(payloadData)
            setLiveTrace((prev) => [...prev, payloadData])
            continue
          }

          if (eventName === 'completed') {
            const result = payloadData?.result ?? {}
            const nextSessionId = payloadData?.session_id ?? null
            if (includeHistory && nextSessionId) {
              setSessionId(nextSessionId)
            }

            appendMessage({
              id: `assistant-${Date.now()}`,
              role: 'assistant',
              text: result.answer || 'No answer returned by backend.',
              sources: Array.isArray(result.sources) ? result.sources : [],
              trace: traces,
            })
            completed = true
            continue
          }

          if (eventName === 'error') {
            const message = payloadData?.message || 'Streaming request failed.'
            throw new Error(message)
          }
        }
      }

      if (!completed) {
        throw new Error('Stream ended before completion event.')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reach backend.')
      appendMessage({
        id: `assistant-error-${Date.now()}`,
        role: 'assistant',
        text: 'I could not reach the backend. Check that FastAPI is running on port 8000.',
        sources: [],
        trace: [],
      })
    } finally {
      setLiveTrace([])
      setIsLoading(false)
    }
  }

  const handleSubmit = async (event) => {
    event.preventDefault()
    await sendQuestion(input)
  }

  return (
    <Box className="app-shell">
      <AppBar position="static" elevation={0} color="transparent" className="top-bar">
        <Toolbar className="top-toolbar">
          <Stack direction="row" spacing={1.5} alignItems="center">
            <Avatar className="shield-avatar">
              <SecurityRoundedIcon fontSize="small" />
            </Avatar>
            <Box>
              <Typography variant="h5" component="h1" className="title-text">
                NIST CSF Copilot
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Chat with your backend about Cybersecurity Framework 2.0
              </Typography>
            </Box>
          </Stack>

          <Stack direction="row" spacing={1.5} alignItems="center">
            <Stack direction="row" spacing={0.5} alignItems="center">
              <Typography variant="body2">Include history in this chat</Typography>
              <Tooltip title={historyTooltipText}>
                <InfoOutlinedIcon fontSize="small" color="action" />
              </Tooltip>
              <Switch checked={includeHistory} onChange={handleHistoryToggle} color="secondary" />
            </Stack>
            <Tooltip title="Start a fresh frontend chat and new backend session id">
              <Button
                variant="outlined"
                color="secondary"
                startIcon={<RestartAltRoundedIcon />}
                onClick={handleReset}
              >
                Reset
              </Button>
            </Tooltip>
          </Stack>
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" className="content-wrap">
        <Card elevation={0} className="control-card">
          <CardContent>
            <Stack spacing={2}>
              <Typography variant="body2" color="text.secondary">
                {historyLabel}
              </Typography>

              <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
                {STARTER_PROMPTS.map((prompt) => (
                  <Chip
                    key={prompt}
                    label={prompt}
                    clickable
                    onClick={() => {
                      setInput(prompt)
                    }}
                    className="starter-chip"
                  />
                ))}
              </Stack>

              {error ? <Alert severity="error">{error}</Alert> : null}
            </Stack>
          </CardContent>
        </Card>

        <Paper elevation={0} className="chat-panel" ref={listRef}>
          <Stack spacing={2.5} className="message-list">
            {messages.map((message) => {
              const isUser = message.role === 'user'
              return (
                <Box key={message.id} className={`message-row ${isUser ? 'user' : 'assistant'}`}>
                  <Paper elevation={0} className={`message-bubble ${isUser ? 'user' : 'assistant'}`}>
                    <Typography variant="body1" className="message-text">
                      {message.text}
                    </Typography>
                    {!isUser && message.sources?.length ? (
                      <Box className="sources-wrap">
                        <Typography variant="caption" className="sources-title">
                          Sources
                        </Typography>
                        <Stack direction="row" spacing={0.75} useFlexGap flexWrap="wrap">
                          {message.sources.slice(0, 5).map((source, index) => (
                            <Chip
                              size="small"
                              key={`${source.source_file}-${source.row_index}-${index}`}
                              label={`${source.source_file}#${source.row_index}`}
                              className="source-chip"
                            />
                          ))}
                        </Stack>
                      </Box>
                    ) : null}
                    {!isUser && message.trace?.length ? (
                      <Box className="trace-wrap">
                        <Typography variant="caption" className="sources-title">
                          Backend Trace
                        </Typography>
                        <TraceTimeline
                          items={message.trace}
                          idPrefix={`trace-${message.id}`}
                        />
                      </Box>
                    ) : null}
                  </Paper>
                </Box>
              )
            })}

            {isLoading ? (
              <Box className="message-row assistant">
                <Paper elevation={0} className="message-bubble assistant loading-bubble">
                  <Stack direction="row" spacing={1.2} alignItems="center">
                    <CircularProgress size={16} color="secondary" />
                    <Typography variant="body2">Thinking about your NIST CSF question...</Typography>
                  </Stack>
                  {liveTrace.length ? (
                    <Box className="trace-wrap live-trace-wrap">
                      <Typography variant="caption" className="sources-title">
                        Live Backend Trace
                      </Typography>
                      <TraceTimeline
                        items={liveTrace}
                        idPrefix="live-trace"
                        defaultOpen
                      />
                    </Box>
                  ) : null}
                </Paper>
              </Box>
            ) : null}
          </Stack>
        </Paper>

        <Paper component="form" elevation={0} className="composer" onSubmit={handleSubmit}>
          <TextField
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Ask about governance, profiles, implementation tiers, controls mapping..."
            multiline
            minRows={1}
            maxRows={5}
            fullWidth
            disabled={isLoading}
          />
          <IconButton
            className="send-button"
            type="submit"
            color="secondary"
            disabled={isLoading || !input.trim()}
          >
            <SendRoundedIcon />
          </IconButton>
        </Paper>
      </Container>
    </Box>
  )
}

export default App
