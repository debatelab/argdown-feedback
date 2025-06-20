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

flowchart LR
    A[Problem<br/>statement] -->|LLM| B@{ shape: docs, label: "Solutions<br/>S<sub>1</sub>....S<sub>k</sub>"}
    B -->|VR| E@{ shape: docs, label: "Evaluations<br/>E<sub>1</sub>...E<sub>k</sub>"} --> C{Correct?}
    C -->|None| SC(Self-Critique)
    C -->|Some or All| H(HIRPO)
    H --> PP
    SC --> PP@{ shape: docs, label: "Pref pairs"}



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

flowchart LR
    OS[Solution S<sub>i</sub><br/>_–incorrect–_]
    OE[Evaluation E<sub>i</sub>]
    A(LLM<br/>Feedback<br/>Prompt)
    B(LLM<br/>Revision<br/>Prompt)
    F@{ shape: docs, label: "Feedbacks<br/>F<sub>1</sub>...F<sub>k</sub>"}
    R@{ shape: docs, label: "Revisions<br/>R<sub>11</sub>...R<sub>1l</sub><br/>...<br/>R<sub>k1</sub>...R<sub>kl</sub>"}
    ER@{ shape: docs, label: "Evaluations<br/>E<sub>11</sub>...E<sub>1l</sub><br/>...<br/>E<sub>k1</sub>...E<sub>kl</sub>"}
    FE(Feedback<br/>Effectiveness)
    HIRPO(HIRPO)
    PP[Preference<br/>Pairs]
    OS --> A
    OE --> A
    A --> F --> B
    B -->R -->|VR| ER
    ER --> FE --> PP
    ER --> HIRPO --> PP
    R -.->|ranked by| HIRPO
    F -.->|ranked by| FE



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

### DAG


```mermaid
flowchart TB

%%Top comprehensive tasks%%


T1{{➡ArgMap+ArgAnno+LogReco}}

T2{{➡ArgMap+ArgAnno}}
T3{{➡ArgAnno+LogReco}}
T4{{➡ArgMap+LogReco}}
T9{{➡ArgAnno+InfReco}}
T10{{➡ArgMap+InfReco}}

T5((➡ArgMap))
T6((➡ArgAnno))
T7((➡LogReco))
T8((➡InfReco))

T1 --> T2 
T1 --> T3 --> T9
T1 --> T4 --> T10

T2 --> T5
T2 --> T6
T3 --> T6
T3 --> T7
T4 --> T5
T4 --> T7
T9 --> T6
T9 --> T8
T10 --> T5
T10 --> T8

SE1[[ArgAnno➡ArgMap]]
T2 --> SE1 --> T6

SE2[[ArgMap➡ArgAnno]]
T2 --> SE2 --> T5

%%Logical reconstruction exercises%%

SE3[[ArgAnno➡InfReco]]
T8 --> SE3 --> T6


SE4[[InfReco➡LogReco]]
T7 --> SE4 --> T8

classDef Completed fill:#292;
class T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,SE1,SE2,SE3,SE4 Completed;


```
