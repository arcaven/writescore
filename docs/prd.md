# WriteScore Product Requirements Document

**Version:** 1.0
**Status:** Baseline (Reverse-Engineered from Existing Codebase)
**Last Updated:** 2025-11-26
**Product Version:** 6.3.0

---

## Change Log

| Change | Date | Version | Description | Author |
|--------|------|---------|-------------|--------|
| Initial PRD | 2025-11-26 | 1.0 | Reverse-engineered PRD from existing codebase v6.3.0 | PM Agent |

---

## 1. Project Overview

### 1.1 Analysis Source

- **Analysis Method:** IDE-based fresh analysis of existing codebase
- **Repository:** `/Users/jmagady/Dev/writescore`
- **Codebase Version:** 6.3.0

### 1.2 Current Project State

**WriteScore** is an AI writing pattern analysis and scoring tool that examines text documents—primarily Markdown—to identify AI-generated content patterns and provide actionable feedback for improving writing quality.

**Primary Purpose:**
- Detect AI-generated writing patterns across multiple linguistic dimensions
- Score documents on a dual scale: **Detection Risk** (0-100, lower = better) and **Quality Score** (0-100, higher = better)
- Provide line-by-line detailed diagnostics with specific improvement suggestions

**Core Capabilities:**
1. **Multi-Dimensional Analysis** - 18 dimension analyzers including perplexity, burstiness, formatting (em-dash detection), voice, semantic coherence, figurative language, etc.
2. **Tier-Based Weighting** - Dimensions classified into ADVANCED (ML-based), CORE (proven AI signatures), SUPPORTING (quality indicators), and STRUCTURAL
3. **Analysis Modes** - FAST (5-15s), ADAPTIVE (30-240s, default), SAMPLING (configurable), FULL (5-20min)
4. **History Tracking** - Track optimization journey over multiple editing iterations
5. **Parameter Recalibration** - Derive scoring parameters from validation datasets
6. **Version Management** - Deploy, rollback, and diff parameter versions

### 1.3 Target Users

**Primary Users:** Individual writers improving their own content (all writing types)
- Technical writers
- Fiction authors
- General content creators
- Any writer wanting to reduce AI-detectable patterns

**Product Positioning:** Personal writing tool (SaaS potential for future consideration)

### 1.4 Goals

- **Detect AI-generated writing patterns** across multiple linguistic dimensions with high accuracy
- **Provide actionable feedback** with specific line-by-line suggestions for improving writing authenticity
- **Score documents objectively** using a dual-score system that separates detection risk from writing quality
- **Support iterative improvement** by tracking score history across editing sessions
- **Scale to book-length documents** through intelligent sampling modes that balance speed vs. accuracy
- **Enable customization** via domain-specific terms, dimension profiles, and recalibrated parameters
- **Maintain scientific rigor** through percentile-anchored scoring derived from validated human/AI datasets

### 1.5 Background Context

**Why WriteScore Exists:**

As AI writing assistants (ChatGPT, Claude, etc.) have become ubiquitous, writers face a new challenge: their AI-assisted content often contains detectable patterns that undermine authenticity and trigger AI detection tools. These patterns include overuse of em-dashes, formulaic transitions ("Furthermore," "Moreover,"), uniform sentence lengths, and predictable vocabulary choices.

WriteScore addresses this by providing writers with detailed analysis of their text across 18 linguistic dimensions—far more comprehensive than simple AI detectors that only give a binary "AI/human" verdict. Rather than just flagging content as AI-generated, WriteScore shows *exactly which patterns* trigger detection and *how to fix them*, enabling writers to maintain their authentic voice while leveraging AI assistance.

**Problem It Solves:**

The tool fills a gap between crude AI detectors (which only accuse) and expensive human editing (which doesn't scale). By providing granular, dimension-by-dimension feedback with specific replacement suggestions, WriteScore empowers individual writers to self-edit their AI-assisted content into genuinely human-sounding prose—whether they're writing technical documentation, fiction, or general content.

---

## 2. Requirements

### 2.1 Functional Requirements

| ID | Requirement |
|----|-------------|
| **FR1** | The system shall analyze Markdown text documents and produce scores across multiple linguistic dimensions |
| **FR2** | The system shall calculate a dual score: Detection Risk (0-100, lower=better) and Quality Score (0-100, higher=better) |
| **FR3** | The system shall support four analysis modes: FAST (truncated), ADAPTIVE (smart sampling), SAMPLING (configurable), and FULL (complete) |
| **FR4** | The system shall provide detailed line-by-line diagnostics with context, problem identification, and specific replacement suggestions |
| **FR5** | The system shall detect AI vocabulary patterns including "delve," "robust," "leverage," "harness," and 30+ other AI-typical words |
| **FR6** | The system shall detect formatting patterns including em-dash overuse (10x more common in AI text than human) |
| **FR7** | The system shall measure sentence burstiness (variation in sentence length) as a key human-vs-AI discriminator |
| **FR8** | The system shall track score history across multiple editing iterations for a single document |
| **FR9** | The system shall support batch analysis of all .md files in a directory |
| **FR10** | The system shall output results in text, JSON, or TSV formats |
| **FR11** | The system shall allow recalibration of scoring parameters from validation datasets |
| **FR12** | The system shall support parameter version management (deploy, rollback, diff, versions) |
| **FR13** | The system shall provide dimension profiles (fast/balanced/full) to control which dimensions are loaded |
| **FR14** | The system shall allow custom domain-specific terms to be configured for technical writing analysis |
| **FR15** | The system shall strip HTML comments (metadata blocks) before analysis to avoid false positives |

### 2.2 Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| **NFR1** | FAST mode shall complete analysis in 5-15 seconds for any document size |
| **NFR2** | ADAPTIVE mode shall complete analysis in 30-240 seconds for book-chapter-length documents (~90 pages) |
| **NFR3** | FULL mode may take 5-20 minutes for large documents but shall analyze 100% of content |
| **NFR4** | The system shall support Python 3.9, 3.10, 3.11, and 3.12 |
| **NFR5** | The system shall be installable via `pip install -e .` with optional dependency groups (dev, ml) |
| **NFR6** | The scoring convention shall use 0-100 scale where 100 = most human-like consistently across all dimensions |
| **NFR7** | The system shall provide meaningful results for documents as short as 50 characters |
| **NFR8** | The system shall handle documents up to 500,000+ characters (book-length) via sampling modes |
| **NFR9** | Dimension analyzers shall self-register via the DimensionRegistry pattern for extensibility |
| **NFR10** | The CLI shall provide interactive confirmation for FULL mode on large documents (>500k chars) |

### 2.3 Compatibility Requirements

| ID | Requirement |
|----|-------------|
| **CR1** | The system shall maintain backward compatibility with existing `.ai-analysis-history` files |
| **CR2** | The system shall support existing parameter file formats (JSON and YAML) |
| **CR3** | CLI command structure shall remain stable (`writescore analyze`, `writescore recalibrate`) |
| **CR4** | Dimension output format shall remain compatible with existing history tracking and reporting |

---

## 3. Technical Constraints and Integration

### 3.1 Technology Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.9+ (supports 3.9, 3.10, 3.11, 3.12) |
| **Package Manager** | pip with setuptools (pyproject.toml) |
| **CLI Framework** | Click 8.0+ |
| **NLP Libraries** | NLTK 3.8+, spaCy 3.7+, textstat 0.7.3+ |
| **ML/Transformers** | transformers 4.35+, torch 2.0+, sentence-transformers 2.0+ |
| **Text Processing** | marko 2.0+ (Markdown parsing), textacy 0.13+ |
| **Data/Math** | numpy 1.24+, scipy 1.11+ |
| **Testing** | pytest 7.4+, pytest-cov, pytest-timeout |
| **Linting** | ruff 0.1+ |
| **Database** | None (file-based history in `.ai-analysis-history/`) |
| **Infrastructure** | Local CLI tool (no server infrastructure) |

**Optional Dependency Groups:**
- `[dev]` - pytest, pytest-cov, pytest-timeout, psutil, ruff, sentence-transformers
- `[ml]` - accelerate 0.20+, datasets 2.14+

> **Note:** `sentence-transformers` is required for the `semantic_coherence` dimension and should be a core dependency.

### 3.2 Code Organization

```
src/writescore/
├── __init__.py          # Package init with version
├── cli/                 # Click-based CLI
│   ├── main.py          # Entry point, commands
│   └── formatters.py    # Output formatting
├── core/                # Core analysis engine
│   ├── analyzer.py      # AIPatternAnalyzer orchestrator
│   ├── analysis_config.py   # AnalysisConfig, AnalysisMode
│   ├── dimension_registry.py # Thread-safe dimension registry
│   ├── dimension_loader.py   # Lazy dimension loading
│   └── results.py       # Result dataclasses
├── dimensions/          # Self-registering dimension analyzers
│   ├── base_strategy.py # DimensionStrategy base class
│   ├── perplexity.py    # AI vocabulary detection
│   ├── burstiness.py    # Sentence variation
│   └── ... (18 dimensions)
├── scoring/             # Dual score calculation
│   ├── dual_score.py    # DualScore dataclass
│   └── dual_score_calculator.py
├── history/             # Score tracking
│   ├── tracker.py       # ScoreHistory
│   └── trends.py        # Trend visualization
├── data/                # Static data files (JSON, TXT)
├── evidence/            # Evidence extraction
└── utils/               # Utilities
```

### 3.3 Coding Standards

- **Line length:** 100 characters
- **Target:** Python 3.9
- **Naming:** Modules (`snake_case.py`), Classes (`PascalCase`), Functions (`snake_case`), Constants (`UPPER_SNAKE_CASE`)
- **Linting:** ruff with E, F, W, I, UP, B, C4, SIM rules
- **Docstrings:** Google style with type hints
- **Story references:** In code comments (e.g., `# Story 1.4.11: Registry-based dimension loading`)

### 3.4 Deployment and Operations

| Aspect | Current Approach |
|--------|------------------|
| **Build Process** | `pip install -e .` for development; `pip install writescore` for distribution |
| **Entry Point** | `writescore` CLI command (defined in pyproject.toml `[project.scripts]`) |
| **Configuration** | `config/scoring_parameters.yaml` for scoring params; CLI flags for runtime config |
| **History Storage** | File-based: `.ai-analysis-history/{document}.history.json` per document |
| **Parameter Versions** | `config/parameters/` with archive in `config/parameters/archive/` |
| **Logging** | stderr for warnings/debug; stdout for results |
| **Testing** | pytest with markers: `slow`, `integration`, `accuracy`; 300s timeout |

### 3.5 Risk Assessment

| Risk Type | Risk | Mitigation |
|-----------|------|------------|
| **Technical** | Transformer models (GLTR) are slow on large documents | ADAPTIVE/SAMPLING modes limit analysis scope |
| **Technical** | spaCy/NLTK require model downloads on first run | Lazy loading with fallback handling |
| **Technical** | Memory usage with large documents + ML models | Sampling modes reduce memory footprint |
| **Dependency** | transformers/torch versions can conflict | Pinned minimum versions in pyproject.toml |
| **Compatibility** | Python 3.9 approaching EOL (Oct 2025) | Already supports 3.10-3.12 |

---

## 4. Feature Inventory

### 4.1 Dimension Analyzers

| Dimension | Purpose | Tier |
|-----------|---------|------|
| `perplexity` | AI vocabulary detection | CORE |
| `burstiness` | Sentence/paragraph variation | CORE |
| `formatting` | Em-dash patterns (strongest AI signal) | CORE |
| `voice` | First-person, contractions, authenticity | CORE |
| `ai_vocabulary` | AI-specific word patterns | CORE |
| `transition_marker` | Formulaic transition detection | CORE |
| `predictability` | GLTR/n-gram analysis | ADVANCED |
| `semantic_coherence` | Cross-sentence coherence | ADVANCED |
| `syntactic` | Dependency tree complexity | SUPPORTING |
| `lexical` | Type-token ratio diversity | SUPPORTING |
| `figurative_language` | Similes, metaphors, idioms | SUPPORTING |
| `sentiment` | Emotional variance detection | SUPPORTING |
| `readability` | Flesch-Kincaid, Gunning Fog | SUPPORTING |
| `advanced_lexical` | MTLD, stemmed diversity | SUPPORTING |
| `pragmatic_markers` | Discourse markers | SUPPORTING |
| `structure` | Heading hierarchy, list patterns | STRUCTURAL |

### 4.2 CLI Commands

| Command | Purpose |
|---------|---------|
| `writescore analyze FILE` | Main analysis command |
| `writescore analyze --batch DIR` | Batch directory analysis |
| `writescore analyze --detailed` | Line-by-line diagnostics |
| `writescore analyze --show-scores` | Dual score with optimization path |
| `writescore analyze --mode MODE` | Select analysis mode (fast/adaptive/sampling/full) |
| `writescore recalibrate DATASET` | Derive parameters from validation data |
| `writescore versions` | List parameter versions |
| `writescore rollback --version V` | Restore previous parameters |
| `writescore diff OLD NEW` | Compare parameter versions |
| `writescore deploy FILE` | Deploy new parameters |

### 4.3 Analysis Modes

| Mode | Speed | Coverage | Use Case |
|------|-------|----------|----------|
| **FAST** | 5-15s | 1-5% | Quick drafts, previews |
| **ADAPTIVE** | 30-240s | 10-20% | Book chapters (recommended) |
| **SAMPLING** | 60-300s | Custom | Specific requirements |
| **FULL** | 5-20min | 100% | Final validation |

---

## 5. Epic and Story Structure

### 5.1 Epic Overview

| Epic | Title | Status |
|------|-------|--------|
| **1.x** | Foundation & Dimension Architecture | Completed |
| **2.x** | Advanced Dimensions & Scoring | Completed |
| **3.x** | Content-Aware Analysis | Planned |
| **4.x** | Repository Extraction | Completed |
| **5.x** | README Modernization | In Progress |

### 5.2 Epic 1: Foundation & Dimension Architecture (Completed)

**Goal:** Establish the core analysis architecture with self-registering dimensions and registry pattern.

| Story | Title |
|-------|-------|
| 1.1 | Enhanced Dimension Base |
| 1.2 | Dimension Registry |
| 1.3 | Weight Validation Mediator |
| 1.4 | Refactor Existing Dimensions |
| 1.4.5 | Split Multi-Concern Dimensions |
| 1.5 | Evidence Extraction |
| 1.10 | Dynamic Reporting |
| 1.16 | Fix Dynamic Reporting Architecture |
| 1.17 | Rename to WriteScore |

### 5.3 Epic 2: Advanced Dimensions & Scoring (Completed)

**Goal:** Add sophisticated linguistic dimensions and improve scoring accuracy.

| Story | Title |
|-------|-------|
| 2.1 | Figurative Language Dimension |
| 2.2 | Pragmatic Markers Dimension |
| 2.3 | Semantic Coherence Dimension |
| 2.4 | Dimension Scoring Optimization |
| 2.4.0.6 | Extract AI Vocabulary Dimension |
| 2.4.0.7 | Implement True Perplexity |
| 2.5 | Percentile-Anchored Scoring |
| 2.6 | Expand Pragmatic Markers Lexicon |
| 2.9 | Short Content Optimization |

### 5.4 Epic 3: Content-Aware Analysis (Planned)

**Goal:** Adapt analysis based on content type (technical, fiction, academic, etc.)

| Story | Title | Status |
|-------|-------|--------|
| 3.1 | Content Type Detection | Planned |
| 3.2 | Content-Aware Dimension Weighting | Planned |
| 3.3 | Content-Aware Scoring Thresholds | Planned |
| 3.4 | Register Consistency Dimension | Planned |
| 3.5 | Person Consistency Dimension | Planned |
| 3.6 | Lexical Inflation Dimension | Planned |
| 3.7 | Content Function Word Ratio | Planned |
| 3.8 | Formulaic Language Appropriateness | Planned |

### 5.5 Epic 4: Repository Extraction (Completed)

**Goal:** Extract WriteScore into standalone GitHub repository with CI/CD.

| Story | Title |
|-------|-------|
| 4.1 | Repository Setup & Source Migration |
| 4.2 | Test & Documentation Migration |
| 4.3 | CI/CD & Release Automation |

### 5.6 Epic 5: README Modernization (In Progress)

**Goal:** Transform README into user-centric, 2025 best-practices documentation.

| Story | Title | Status |
|-------|-------|--------|
| 5.0 | README Modernization Epic | In Progress |
| 5.1 | Modernize README 2025 Best Practices | In Progress |

---

## 6. Future Enhancements

Based on Epic 3 planning and codebase analysis, the following enhancements are under consideration:

1. **Content-Aware Analysis** - Automatically detect content type and adjust dimension weights/thresholds
2. **Web Interface** - Optional web UI for non-CLI users
3. **SaaS Offering** - Hosted service for broader accessibility
4. **API Endpoint** - REST API for integration with writing tools
5. **IDE Plugins** - VS Code, Obsidian extensions for real-time feedback
6. **Additional File Formats** - Support for .txt, .docx, .rst beyond Markdown

---

## Appendix A: Quick Reference

### Installation

```bash
# Development install
pip install -e ".[dev]"

# Run analysis
writescore analyze document.md

# Detailed diagnostics
writescore analyze document.md --detailed

# Dual scoring
writescore analyze document.md --show-scores
```

### Scoring Convention

- **0-100 scale** where **100 = most human-like**
- **Detection Risk:** Lower is better (target: <30)
- **Quality Score:** Higher is better (target: >85)

### Dimension Tiers

| Tier | Weight | Examples |
|------|--------|----------|
| ADVANCED | 30-40% | predictability, semantic_coherence |
| CORE | 35-45% | burstiness, formatting, voice, perplexity |
| SUPPORTING | 15-25% | lexical, sentiment, readability |
| STRUCTURAL | 5-10% | structure |

---

*This PRD was reverse-engineered from the WriteScore v6.3.0 codebase to establish a baseline product requirements document.*
