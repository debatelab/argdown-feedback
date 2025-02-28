# argdown-hirpo

Hindsight Instruction Relabeling Preferences

### Main preference generation workflow

```mermaid 
flowchart LR
    A[Problem<br/>statement] -->|LLM| B@{ shape: docs, label: "Candidate<br/>Solutions S"}
    B -->|Judge| E[Evals _E_] --> C{Valid?}
    C -->|None| SC[Self-Critique]
    C -->|Some or All| Router{R}
    Router -->|p| HIR2[HIRP<br/>Validity]
    Router -->|1-p| HIR3[HIRP<br/>Virtue]
    SC --> PP@{ shape: docs, label: "Pref pairs"}
    HIR2 --> PP 
    HIR3 --> PP 
    PP -->|empty?| HIR1[HIRP<br/>FailureType] 
```

### How does Self-Critique work?

```mermaid
flowchart LR
    A[Invalid<br/>solution _s_]
    B@{ shape: docs, label: "Feedback<br/>candidates<br/>_f1_ ... _fk_"}
    C@{ shape: docs, label: "Candidate<br/>revisions<br/>_r11_, ... _r1l_<br/>...<br/>_rk1_ ... _rkl_"}
    A -->|LLM| B -->|LLM| C -->|Judge| E[Evals _E_]
    E --> HIR[Solution HIRP<br/>for _ri1_ ... _ril_<br/>with i=1...k] 
    HIR --> SCP[Feedback<br/>Preferences]
    C -.->|ranked by| HIR
    B -.->|ranked by| SCP
```

### Symmetric HIRP (illustration of key idea)

```markdown
1. Problem statement: p
2. Candidate solutions: s1, s2
3. Judge: s1 valid, s2 invalid
4. HIRP pairs:
   - prompt: Solve p!
     chosen: s1
     rejected s2
   - prompt: Present **invalid** solution to p!
     chosen: s2
     rejected: s1
```
