import http from 'node:http';
import express from 'express';
import type { Request, Response } from 'express';
import path from 'path';
import dotenv from 'dotenv';
import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';

dotenv.config();

const app = express();
const PORT = Number(process.env.PORT) || 3000;

app.use(express.json({ limit: '1mb' }));

const SYSTEM_INSTRUCTION = `You are an advanced Computer Science Research Agent specializing in parsing academic papers, extracting system architectures, and identifying student development opportunities.
- Prioritize token efficiency and keep total analysis concise and well under 25,000 tokens.
- When given a research paper title, abstract, or URL, execute this exact 4-stage Chain-of-Thought pipeline:

Step 1 - Core Concept Extraction:
Summarize the problem statement, the primary methodology introduced, and the key mathematical or algorithmic breakthroughs in under 300 words total using plain, accessible language.

Step 2 - Architectural Flowchart (Mermaid.js):
Generate a clean, syntactically valid Mermaid.js flowchart starting with "graph TD" that charts the components, data inputs, model layers, and data outputs of the system described in the paper.
CRITICAL MERMAID RULES:
- Always start the string with "graph TD" on line 1.
- Use simple node IDs (A, B, C, D, E, F, G, H) and wrap EVERY node label in double quotes inside square brackets, e.g., A["Input Tokens + Positional Encoding"] --> B["Multi-Head Self-Attention"].
- Never put unescaped double quotes inside a label; use plain words, +, -, /, or parentheses inside the quoted label.
- Keep 6 to 10 meaningful nodes tracing input -> core modules -> output.
- Do not include markdown code fences (\`\`\`) inside the flowchartMermaid field.

Step 3 - Future Work & Student Project Opportunities:
Brainstorm exactly 3 concrete, realistic ways a 3rd-year Computer Science student could build upon, extend, or optimize this paper for a resume or internship project. For each idea provide:
- A concise project title
- Exact Extension (specific, actionable engineering build)
- Targeted Performance Metric (concrete measurable target like latency reduction, memory footprint, or accuracy trade-off)
- Recommended Tech Stack (3-5 real libraries/tools like PyTorch, ONNX Runtime, Hugging Face, FastAPI, React, Triton, FAISS)

Step 4 - Paper Metadata:
Provide accurate bibliographic metadata:
- citations: integer estimate or verified citation count (non-negative integer)
- authors: comma-separated author string
- publication: conference/journal/preprint venue (e.g., NeurIPS, ICLR, CVPR, arXiv)
- year: 4-digit integer publication year`;

const ANALYSIS_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    paperTitle: {
      type: Type.STRING,
      description: 'The canonical title of the research paper.',
    },
    coreConcept: {
      type: Type.OBJECT,
      properties: {
        problemStatement: {
          type: Type.STRING,
          description: 'Plain-language summary of the core problem the paper solves (1-3 sentences).',
        },
        primaryMethodology: {
          type: Type.STRING,
          description: 'Plain-language summary of the primary methodology or architecture introduced (1-3 sentences).',
        },
        keyBreakthroughs: {
          type: Type.STRING,
          description: 'Key mathematical or algorithmic breakthroughs explained clearly (1-3 sentences).',
        },
        summary: {
          type: Type.STRING,
          description: 'Unified accessible synthesis under 200 words combining problem, method, and breakthrough.',
        },
      },
      required: ['problemStatement', 'primaryMethodology', 'keyBreakthroughs', 'summary'],
    },
    flowchartMermaid: {
      type: Type.STRING,
      description: 'Valid Mermaid.js graph TD flowchart using quoted node labels like A["Input"] --> B["Layer"]. No markdown fences.',
    },
    studentProjects: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          title: {
            type: Type.STRING,
            description: 'Short resume-ready project title.',
          },
          extension: {
            type: Type.STRING,
            description: 'Exact engineering extension a 3rd-year CS student can build.',
          },
          metric: {
            type: Type.STRING,
            description: 'Targeted performance metric (e.g., 3x inference speedup & 60% RAM reduction).',
          },
          techStack: {
            type: Type.ARRAY,
            items: { type: Type.STRING },
            description: '3 to 5 recommended technologies/libraries.',
          },
        },
        required: ['title', 'extension', 'metric', 'techStack'],
      },
    },
    metadata: {
      type: Type.OBJECT,
      properties: {
        citations: {
          type: Type.INTEGER,
          description: 'Estimated or verified citation count as an integer.',
        },
        authors: {
          type: Type.STRING,
          description: 'Comma-separated list of primary authors.',
        },
        publication: {
          type: Type.STRING,
          description: 'Publication venue (e.g. NeurIPS, ICML, ICLR, CVPR, arXiv).',
        },
        year: {
          type: Type.INTEGER,
          description: '4-digit publication year.',
        },
      },
      required: ['citations', 'authors', 'publication', 'year'],
    },
  },
  required: ['paperTitle', 'coreConcept', 'flowchartMermaid', 'studentProjects', 'metadata'],
};

function sanitizeMermaidGraph(raw: string): string {
  let cleaned = (raw || '')
    .replace(/```mermaid/gi, '')
    .replace(/```/g, '')
    .replace(/\[FLOWCHART\]/gi, '')
    .trim();

  if (!/^(graph|flowchart)\s+(TD|TB|LR|RL|BT)/i.test(cleaned)) {
    cleaned = `graph TD\n${cleaned}`;
  }

  const lines = cleaned.split(/\r?\n/).map((line) => {
    const trimmed = line.trim();
    if (!trimmed || /^(graph|flowchart)\s+/i.test(trimmed) || /^subgraph\b/i.test(trimmed) || /^end$/i.test(trimmed)) {
      return trimmed;
    }
    // Ensure unquoted [Label] blocks are wrapped in double quotes for safe Mermaid parsing
    return trimmed.replace(/(\b[A-Za-z0-9_-]+)\[(?!["'])([^\]\n]+)\]/g, (_m, id: string, label: string) => {
      const safeLabel = label.replace(/"/g, "'").trim();
      return `${id}["${safeLabel}"]`;
    });
  });

  return lines.filter(Boolean).join('\n');
}

function extractArxivId(input: string): string | null {
  const urlMatch = input.match(/arxiv\.org\/(?:abs|pdf|html)\/(\d{4}\.\d{4,5}(?:v\d+)?)/i);
  if (urlMatch) return urlMatch[1];
  const bareMatch = input.trim().match(/^(?:arXiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)$/i);
  if (bareMatch) return bareMatch[1];
  return null;
}

async function fetchArxivMetadataContext(arxivId: string): Promise<string | null> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 3500);
    const res = await fetch(`https://export.arxiv.org/api/query?id_list=${encodeURIComponent(arxivId)}`, {
      signal: controller.signal,
    });
    clearTimeout(timer);
    if (!res.ok) return null;
    const xml = await res.text();
    const entryMatch = xml.match(/<entry>([\s\S]*?)<\/entry>/);
    if (!entryMatch) return null;
    const entry = entryMatch[1];
    const title = entry.match(/<title>([\s\S]*?)<\/title>/)?.[1]?.replace(/\s+/g, ' ').trim();
    const summary = entry.match(/<summary>([\s\S]*?)<\/summary>/)?.[1]?.replace(/\s+/g, ' ').trim();
    const published = entry.match(/<published>([\s\S]*?)<\/published>/)?.[1]?.trim();
    const authorMatches = [...entry.matchAll(/<author>\s*<name>([\s\S]*?)<\/name>/g)].map((m) =>
      m[1].replace(/\s+/g, ' ').trim()
    );
    if (!title) return null;
    return [
      `Verified arXiv Record (${arxivId}):`,
      `Title: ${title}`,
      authorMatches.length ? `Authors: ${authorMatches.join(', ')}` : '',
      published ? `Published: ${published.slice(0, 4)}` : '',
      summary ? `Abstract: ${summary}` : '',
    ]
      .filter(Boolean)
      .join('\n');
  } catch {
    return null;
  }
}

function countWords(text: string): number {
  const words = text.trim().split(/\s+/).filter(Boolean);
  return words.length;
}

app.get('/api/health', (_req: Request, res: Response) => {
  res.json({ status: 'ok' });
});

app.post('/api/analyze', async (req: Request, res: Response) => {
  try {
    const rawQuery = typeof req.body?.query === 'string' ? req.body.query.trim() : '';
    if (!rawQuery) {
      res.status(400).json({ error: 'Please provide a research paper title, URL, or abstract.' });
      return;
    }

    if (rawQuery.length > 15000) {
      res.status(400).json({ error: 'Input is too long. Please keep paper text under 15,000 characters.' });
      return;
    }

    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      res.status(500).json({
        error: 'GEMINI_API_KEY is not configured. Please check your API key in Settings > Secrets.',
      });
      return;
    }

    const ai = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });

    const isUrl = /^https?:\/\/\S+$/i.test(rawQuery);
    const arxivId = extractArxivId(rawQuery);
    const arxivContext = arxivId ? await fetchArxivMetadataContext(arxivId) : null;

    const userPrompt = arxivContext
      ? `Analyze the following research paper and execute the 4-stage pipeline:\n\nUser Input: ${rawQuery}\n\n${arxivContext}`
      : `Analyze the following research paper (title, URL, or abstract) and execute the 4-stage pipeline:\n\n${rawQuery}`;

    const useGrounding = !arxivContext && (isUrl || rawQuery.length < 300);
    const tools = useGrounding
      ? isUrl
        ? [{ urlContext: {} }, { googleSearch: {} }]
        : [{ googleSearch: {} }]
      : undefined;

    let response;
    try {
      response = await ai.models.generateContent({
        model: 'gemini-3.8-flash',
        contents: userPrompt,
        config: {
          systemInstruction: SYSTEM_INSTRUCTION,
          thinkingConfig: { thinkingLevel: ThinkingLevel.MEDIUM },
          responseMimeType: 'application/json',
          responseSchema: ANALYSIS_SCHEMA,
          ...(tools ? { tools } : {}),
        },
      });
    } catch (firstErr) {
      // If grounding tools + responseSchema conflict on a request, retry cleanly without tools
      if (tools) {
        response = await ai.models.generateContent({
          model: 'gemini-3.8-flash',
          contents: userPrompt,
          config: {
            systemInstruction: SYSTEM_INSTRUCTION,
            thinkingConfig: { thinkingLevel: ThinkingLevel.MEDIUM },
            responseMimeType: 'application/json',
            responseSchema: ANALYSIS_SCHEMA,
          },
        });
      } else {
        throw firstErr;
      }
    }

    const rawText = (response.text ?? '').trim();
    if (!rawText) {
      res.status(502).json({ error: 'Received an empty response from Gemini. Please try again.' });
      return;
    }

    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(rawText);
    } catch {
      const jsonMatch = rawText.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        res.status(502).json({ error: 'Unable to parse structured analysis from model output.' });
        return;
      }
      parsed = JSON.parse(jsonMatch[0]);
    }

    const core = (parsed.coreConcept as Record<string, string>) || {};
    const problemStatement = String(core.problemStatement || '').trim();
    const primaryMethodology = String(core.primaryMethodology || '').trim();
    const keyBreakthroughs = String(core.keyBreakthroughs || '').trim();
    const summary = String(core.summary || '').trim();
    const combinedWordCount = countWords(
      `${problemStatement} ${primaryMethodology} ${keyBreakthroughs} ${summary}`
    );

    const rawProjects = Array.isArray(parsed.studentProjects) ? parsed.studentProjects : [];
    const studentProjects = rawProjects.slice(0, 3).map((item, idx) => {
      const p = (item || {}) as Record<string, unknown>;
      const stack = Array.isArray(p.techStack)
        ? p.techStack.map((t) => String(t).trim()).filter(Boolean)
        : ['PyTorch', 'FastAPI'];
      return {
        id: `proj-${idx + 1}`,
        title: String(p.title || `Student Extension ${idx + 1}`).trim(),
        extension: String(p.extension || '').trim(),
        metric: String(p.metric || '').trim(),
        techStack: stack.slice(0, 5),
      };
    });

    const rawMeta = (parsed.metadata as Record<string, unknown>) || {};
    const citations = Math.max(0, Math.round(Number(rawMeta.citations) || 0));
    const authors = String(rawMeta.authors || 'Unknown Authors').trim();
    const publication = String(rawMeta.publication || 'arXiv').trim();
    const rawYear = Math.round(Number(rawMeta.year) || new Date().getFullYear());
    const year = rawYear >= 1950 && rawYear <= 2030 ? rawYear : new Date().getFullYear();
    const formattedLine = `Metadata: Citations: ${citations}, Authors: ${authors}, Publication: ${publication}, Year: ${year}`;

    const groundingChunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks;
    const groundingSources = Array.isArray(groundingChunks)
      ? groundingChunks
          .map((chunk) => {
            const web = (chunk as { web?: { uri?: string; title?: string } })?.web;
            if (web?.uri) {
              return {
                title: web.title || web.uri,
                uri: web.uri,
              };
            }
            return null;
          })
          .filter((s): s is { title: string; uri: string } => s !== null)
          .slice(0, 5)
      : undefined;

    const totalTokens = response.usageMetadata?.totalTokenCount;

    res.json({
      analysis: {
        paperTitle: String(parsed.paperTitle || rawQuery).trim(),
        inputQuery: rawQuery,
        coreConcept: {
          problemStatement,
          primaryMethodology,
          keyBreakthroughs,
          summary,
          wordCount: combinedWordCount,
        },
        flowchartMermaid: sanitizeMermaidGraph(String(parsed.flowchartMermaid || '')),
        studentProjects,
        metadata: {
          citations,
          authors,
          publication,
          year,
          formattedLine,
        },
        groundingSources: groundingSources && groundingSources.length > 0 ? groundingSources : undefined,
        tokenCount: typeof totalTokens === 'number' ? totalTokens : undefined,
        analyzedAt: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      },
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unexpected server error during paper analysis.';
    const status =
      message.includes('403') || message.includes('PERMISSION_DENIED') || message.includes('API_KEY_INVALID')
        ? 403
        : message.includes('429') || message.includes('RESOURCE_EXHAUSTED')
          ? 429
          : 500;

    res.status(status).json({
      error:
        status === 403
          ? 'Gemini API authentication failed. Please check your GEMINI_API_KEY in Settings > Secrets.'
          : status === 429
            ? 'Gemini API rate limit reached. Please wait a moment and try again.'
            : message,
    });
  }
});

async function startServer() {
  const httpServer = http.createServer(app);

  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        hmr: false,
        ws: { server: httpServer },
        allowedHosts: true,
        watch: null,
      },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('/{*path}', (_req: Request, res: Response) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  httpServer.listen(PORT, '0.0.0.0', () => {
    console.log(`PaperLens server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start PaperLens server:', err);
  process.exit(1);
});
