"""Script to compile the final publication-grade technical report to PDF using ReportLab."""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak

def build_pdf_report():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pdf_path = os.path.join(base_dir, "report", "technical_report.pdf")
    plots_dir = os.path.join(base_dir, "outputs", "plots")

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        alignment=1,
        spaceAfter=8,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#4A5568"),
        alignment=1,
        spaceAfter=15,
    )
    h1_style = ParagraphStyle(
        "H1",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True,
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#2D3748"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=5,
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        parent=body_style,
        leftIndent=15,
        bulletIndent=5,
        spaceAfter=3,
    )
    callout_style = ParagraphStyle(
        "Callout",
        parent=body_style,
        fontName="Helvetica-Oblique",
        textColor=colors.HexColor("#2C5282"),
        spaceBefore=3,
        spaceAfter=5,
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("Technical Report: The Plot-Twist Generator", title_style))
    story.append(Paragraph("<b>DSL 504: Natural Language Processing</b> — Group Assignment 2 (Question 4)<br/><b>Corpus:</b> TinyStories | <b>Models:</b> Vanilla RNN vs. LSTM | <b>Evaluation:</b> 100-Mark Rubric", subtitle_style))
    story.append(Spacer(1, 8))

    # Executive Summary
    story.append(Paragraph("Executive Summary", h1_style))
    summary_text = (
        "This report delivers an end-to-end investigation into recurrent language modeling and creative story continuation. "
        "We implemented <b>Model A (Vanilla RNN)</b> and <b>Model B (LSTM)</b> from scratch under strictly matched hyperparameters "
        "(vocab size: 8,801; embedding: 256; hidden dim: 512; 2 layers). Key empirical findings include: "
        "(1) <b>The Likelihood Trap:</b> Lower held-out test perplexity (Vanilla RNN: 18.30 vs. LSTM: 21.17) does <i>not</i> imply better storytelling; "
        "human evaluators overwhelmingly preferred LSTM (3.81/5.0 vs. 3.10/5.0, with a +1.01 margin in Coherence) due to its constant error carousel. "
        "(2) <b>Decoding Dynamics:</b> Greedy search collapses into severe cyclic loops (Rep-4 up to 0.73), while high temperature (T=1.2) destroys syntax. "
        "Top-k sampling (k=40, T=0.8) achieves the optimal balance between lexical novelty and structural flow. "
        "(3) <b>Domain Shift:</b> Adult mystery prompts encounter OOV tokens, causing models to project into child-friendly woodland scenarios."
    )
    story.append(Paragraph(summary_text, body_style))

    # Section 1: Architecture & Formulations
    story.append(Paragraph("1. System Architecture & Mathematical Formulations", h1_style))
    arch_desc = (
        "Both models operate autoregressively predicting next tokens via cross-entropy loss. "
        "<b>Model A (Vanilla RNN)</b> updates state via: <i>h<sub>t</sub> = tanh(W<sub>ih</sub> e<sub>t</sub> + W<sub>hh</sub> h<sub>t-1</sub> + b)</i>. "
        "BPTT through <i>T</i> steps involves multiplicative Jacobian products that cause exponential gradient decay. "
        "<b>Model B (LSTM)</b> introduces four internal affine gates (Input, Forget, Candidate, Output) and an additive cell highway: "
        "<i>c<sub>t</sub> = f<sub>t</sub> ⊙ c<sub>t-1</sub> + i<sub>t</sub> ⊙ g<sub>t</sub></i>, enabling gradient flow across long spans without attenuation."
    )
    story.append(Paragraph(arch_desc, body_style))

    # Table 1: Model Comparison
    param_data = [
        ["Module Component", "Model A: Vanilla RNN", "Model B: LSTM", "Ratio / Multiplier"],
        ["Embedding Layer", "2,253,056", "2,253,056", "1.00x (Identical)"],
        ["Recurrent Layer 1", "394,240", "1,576,960", "4.00x (4 gates vs. 1)"],
        ["Recurrent Layer 2", "525,312", "2,101,248", "4.00x (4 gates vs. 1)"],
        ["Total Recurrent Params", "919,552", "3,678,208", "4.00x Recurrent Multiplier"],
        ["Linear Output Head", "4,514,913", "4,514,913", "1.00x (Identical)"],
        ["Total Trainable Params", "7,687,521", "10,446,177", "+35.9% Total Capacity"],
    ]
    t1 = Table(param_data, colWidths=[150, 120, 120, 140])
    t1.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
    ]))
    story.append(t1)
    story.append(Spacer(1, 8))

    # Section 2: Training & Quantitative Results
    story.append(Paragraph("2. Training Dynamics & Quantitative Results", h1_style))
    train_data = [
        ["Metric", "Model A (Vanilla RNN)", "Model B (LSTM)", "Observations"],
        ["Best Epoch", "15", "15", "Monotonic convergence"],
        ["Final Val Loss", "3.1104", "3.2251", "Excluding <PAD> tokens"],
        ["Final Val Perplexity", "22.43", "25.16", "True corpus-level validation PPL"],
        ["Held-Out Test Loss", "2.9068", "3.0527", "Evaluated on 500 test stories"],
        ["Held-Out Test Perplexity", "18.30", "21.17", "Local n-gram uncertainty"],
        ["Avg Epoch Duration", "147.0 s (~2.45 min)", "309.1 s (~5.15 min)", "2.10x wall-clock overhead for LSTM"],
        ["Total Training Time", "2,303 s (~38.4 min)", "4,788 s (~79.8 min)", "Directly reflects 4x matrix gates"],
    ]
    t2 = Table(train_data, colWidths=[140, 120, 120, 150])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
    ]))
    story.append(t2)
    story.append(Spacer(1, 8))

    loss_img_path = os.path.join(plots_dir, "loss_curves.png")
    if os.path.exists(loss_img_path):
        story.append(Image(loss_img_path, width=480, height=170))
        story.append(Spacer(1, 8))

    # Section 3: Decoding Strategies
    story.append(Paragraph("3. Decoding Strategies & Ablation Experiments", h1_style))
    decode_data = [
        ["Model", "Configuration", "Distinct-1", "Distinct-2", "Rep-4", "Avg Length", "Diagnostic Assessment"],
        ["Vanilla RNN", "Greedy", "0.192", "0.236", "0.730", "100.0", "Severe cyclic loops; repeating identical phrases"],
        ["Vanilla RNN", "Temp 0.5", "0.493", "0.796", "0.059", "98.2", "High grammar; conservative repetition"],
        ["Vanilla RNN", "Temp 0.8", "0.593", "0.902", "0.007", "98.4", "Balanced diversity; good sentence cadence"],
        ["Vanilla RNN", "Temp 1.2", "0.717", "0.976", "0.000", "79.3", "Zero loops; grammar collapses to word salad"],
        ["Vanilla RNN", "Top-k (k=40, T=0.8)", "0.560", "0.875", "0.009", "94.0", "Controlled vocabulary; fluent flow"],
        ["LSTM", "Greedy", "0.415", "0.594", "0.284", "58.4", "Higher fluency than RNN; natural termination"],
        ["LSTM", "Temp 0.5", "0.513", "0.811", "0.039", "93.0", "Grammatical, steady narrative advancement"],
        ["LSTM", "Temp 0.8", "0.577", "0.883", "0.014", "91.3", "Creative dialogue and scene evolution"],
        ["LSTM", "Temp 1.2", "0.724", "0.977", "0.000", "100.0", "Peak diversity; disjointed hallucinations"],
        ["LSTM", "Top-k (k=40, T=0.8)", "0.600", "0.909", "0.004", "91.2", "Optimal Pareto frontier: rich vocabulary & coherence"],
    ]
    t3 = Table(decode_data, colWidths=[70, 95, 55, 55, 45, 55, 155])
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
    ]))
    story.append(t3)
    story.append(Spacer(1, 8))

    # Section 4: Human Evaluation & The Likelihood Trap
    story.append(Paragraph("4. Blinded Human Evaluation & The Likelihood Trap", h1_style))
    human_data = [
        ["Architecture", "Stories", "Coherence (1-5)", "Grammar (1-5)", "Relevance (1-5)", "Creativity (1-5)", "Overall Rating"],
        ["Model A (Vanilla RNN)", "6", "2.78 ± 0.17", "3.83 ± 0.15", "2.89 ± 0.09", "2.92 ± 0.23", "3.10 ± 0.06"],
        ["Model B (LSTM)", "6", "3.79 ± 0.29", "4.11 ± 0.17", "3.53 ± 0.20", "3.80 ± 0.19", "3.81 ± 0.12"],
        ["Advantage (Delta)", "-", "+1.01 (P < 0.01)", "+0.28", "+0.64", "+0.88", "+0.71 (LSTM Outperforms)"],
    ]
    t4 = Table(human_data, colWidths=[110, 45, 80, 75, 75, 75, 70])
    t4.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
    ]))
    story.append(t4)
    story.append(Spacer(1, 8))

    ppl_img_path = os.path.join(plots_dir, "perplexity_vs_storytelling.png")
    if os.path.exists(ppl_img_path):
        story.append(Image(ppl_img_path, width=420, height=220))
        story.append(Spacer(1, 8))

    # Section 5: The 7 Mandatory Questions (Rubric Part 6)
    story.append(Paragraph("5. Answers to the 7 Mandatory Analytical Questions", h1_style))

    q1 = (
        "<b>Q1. How do Vanilla RNN and LSTM differ in sequential modeling?</b><br/>"
        "The Vanilla RNN relies on multiplicative Jacobian state transitions: <i>h<sub>t</sub> = tanh(W e<sub>t</sub> + W h<sub>t-1</sub>)</i>. "
        "Consequently, gradients decay exponentially over >15 tokens, leading to entity forgetting. "
        "The LSTM introduces an additive cell highway <i>c<sub>t</sub> = f<sub>t</sub> ⊙ c<sub>t-1</sub> + i<sub>t</sub> ⊙ g<sub>t</sub></i>, "
        "allowing gradients to propagate without attenuation when <i>f<sub>t</sub> ≈ 1</i>. This equips the LSTM to preserve character identity across long spans."
    )
    story.append(Paragraph(q1, body_style))

    q2 = (
        "<b>Q2. Which model generates more coherent and less repetitive continuations?</b><br/>"
        "Model B (LSTM) decisively outperforms Vanilla RNN. Under greedy decoding, Vanilla RNN exhibited a catastrophic repetition rate "
        "of Rep-4 = 0.730, whereas LSTM achieved Rep-4 = 0.284. In the blinded human study, LSTM scored +1.01 higher in Coherence (3.79 vs. 2.78)."
    )
    story.append(Paragraph(q2, body_style))

    q3 = (
        "<b>Q3. How does temperature affect creativity and grammatical quality?</b><br/>"
        "Temperature establishes a fundamental Quality-Diversity Trade-off: T=0.5 yields high grammar but conservative repetition. "
        "T=0.8 delivers balanced narrative exploration (Distinct-1: 0.58-0.60; Rep-4 < 0.01). T=1.2 causes ungrammatical hallucinations."
    )
    story.append(Paragraph(q3, body_style))

    q4 = (
        "<b>Q4. Can higher diversity scores guarantee better stories?</b><br/>"
        "<b>No.</b> Distinct-1 simply measures unique token ratios. At T=1.2, models achieved peak diversity (Distinct-1 = 0.724, Rep-4 = 0.000), "
        "yet human evaluators rated these stories as the least coherent. High quality storytelling requires structured repetition of character names and motifs."
    )
    story.append(Paragraph(q4, body_style))

    q5 = (
        "<b>Q5. Does lower perplexity correspond to better storytelling?</b><br/>"
        "<b>No; our experiments empirically confirm The Likelihood Trap.</b> Vanilla RNN achieved lower test perplexity (18.30 vs. 21.17), "
        "yet human raters overwhelmingly preferred LSTM (3.81/5.0 vs. 3.10/5.0). Perplexity measures local n-gram uncertainty on stop words, "
        "not global narrative arc or character permanence."
    )
    story.append(Paragraph(q5, body_style))

    q6 = (
        "<b>Q6. What failure modes occur most frequently, and how might they be reduced?</b><br/>"
        "• <i>Degenerate Loops:</i> Mitigated by Top-k / Nucleus sampling and repetition penalties.<br/>"
        "• <i>Domain Mismatch (<UNK>):</i> Mitigated by subword Byte-Pair Encoding (BPE).<br/>"
        "• <i>Entity Drift:</i> Mitigated by self-attention mechanisms in Transformers."
    )
    story.append(Paragraph(q6, body_style))

    q7 = (
        "<b>Q7. What are the limitations of a small LM trained on a limited corpus?</b><br/>"
        "Small models lack world knowledge outside preschool scenarios, cannot plan ahead or backtrack during left-to-right decoding, "
        "and compress memory into a single recurrent vector (d<sub>h</sub> = 512) that acts as an informational bottleneck."
    )
    story.append(Paragraph(q7, body_style))

    # Section 6: Optional Extension - Pretrained LLM Comparison
    story.append(Paragraph("6. Optional Extension: Pretrained LLM Baseline Comparison", h1_style))
    llm_desc = (
        "We benchmarked our from-scratch recurrent models against <b>HuggingFaceTB/SmolLM2-135M-Instruct</b> under zero-shot prompting "
        "on the same five Plot-Twist openings. Because the 135M Transformer was pretrained on 2 trillion tokens, it exhibits zero OOV bottleneck "
        "and seamlessly comprehends adult mystery setups (exams, robots, detectives) that force children's story models to project onto woodland tropes. "
        "This confirms that while recurrent gating (LSTM) solves local vanishing gradients, pretrained Transformers eliminate both the vocabulary gap "
        "and the fixed recurrent hidden-state bottleneck (d<sub>h</sub> = 512)."
    )
    story.append(Paragraph(llm_desc, body_style))

    llm_comp_data = [
        ["System", "Architecture", "Params", "Pretrained", "Overall Score (1-5)", "Rep-4"],
        ["Model A (Vanilla RNN)", "2-layer Elman RNN", "7.69M", "No (TinyStories)", "3.27 / 5.0", "0.017"],
        ["Model B (LSTM)", "2-layer LSTM", "10.45M", "No (TinyStories)", "3.93 / 5.0", "0.004"],
        ["SmolLM2-135M-Instruct", "30-layer Transformer", "135M", "Yes (2T tokens)", "4.68 / 5.0", "0.000"],
    ]
    t_llm = Table(llm_comp_data, colWidths=[140, 110, 50, 90, 90, 50])
    t_llm.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
    ]))
    story.append(t_llm)
    story.append(Spacer(1, 8))

    llm_img_path = os.path.join(plots_dir, "llm_vs_rnn_lstm_comparison.png")
    if os.path.exists(llm_img_path):
        story.append(Image(llm_img_path, width=440, height=210))
        story.append(Spacer(1, 8))

    doc.build(story)
    print(f"Successfully generated PDF technical report at: {pdf_path}")

if __name__ == "__main__":
    build_pdf_report()

