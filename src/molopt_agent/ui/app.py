"""Gradio UI for interactive molecule optimization."""

import gradio as gr
from dotenv import load_dotenv

from .runner import InteractiveSession, SessionResult, TraceEntry, DEFAULT_TARGET_SMILES
from .mol_utils import smiles_to_image


# Apple-inspired custom CSS
CUSTOM_CSS = """
/* Global styles */
.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI', Roboto, sans-serif !important;
    background-color: #F5F5F7 !important;
    max-width: 100% !important;
    padding: 0 24px !important;
}

/* Main header */
.main-header {
    text-align: center;
    padding: 32px 0 24px 0;
    margin-bottom: 8px;
}

.main-header h1 {
    font-size: 2.5rem !important;
    font-weight: 600 !important;
    color: #1D1D1F !important;
    margin-bottom: 8px !important;
    letter-spacing: -0.02em;
}

.main-header p {
    font-size: 1.1rem !important;
    color: #6E6E73 !important;
    margin: 0 !important;
}

/* Card styling */
.card {
    background: #FFFFFF !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08), 0 4px 12px rgba(0, 0, 0, 0.04) !important;
    border: none !important;
}

.card-header {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #1D1D1F !important;
    margin-bottom: 16px !important;
    padding-bottom: 12px !important;
    border-bottom: 1px solid #E8E8ED !important;
}

/* Form elements */
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    border-radius: 10px !important;
    border: 1px solid #D2D2D7 !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

.gradio-container input:focus,
.gradio-container textarea:focus,
.gradio-container select:focus {
    border-color: #007AFF !important;
    box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.15) !important;
    outline: none !important;
}

/* Labels */
.gradio-container label {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #1D1D1F !important;
}

.gradio-container .info {
    color: #6E6E73 !important;
    font-size: 0.8rem !important;
}

/* Primary button */
.primary-btn {
    background: #007AFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    padding: 12px 24px !important;
    transition: all 0.2s ease !important;
}

.primary-btn:hover {
    background: #0066D6 !important;
    transform: translateY(-1px) !important;
}

/* Secondary button */
.secondary-btn {
    background: #FFFFFF !important;
    border: 1px solid #D2D2D7 !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    color: #1D1D1F !important;
    transition: all 0.2s ease !important;
}

.secondary-btn:hover {
    background: #F5F5F7 !important;
    border-color: #007AFF !important;
}

/* Navigation buttons */
.nav-btn {
    background: #F5F5F7 !important;
    border: 1px solid #E8E8ED !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    color: #1D1D1F !important;
    min-width: 90px !important;
}

.nav-btn:hover {
    background: #E8E8ED !important;
}

/* Sliders */
.gradio-container input[type="range"] {
    accent-color: #007AFF !important;
}

/* Score display */
.score-display {
    text-align: center;
    padding: 12px !important;
    background: #F5F5F7 !important;
    border-radius: 10px !important;
}

.score-display label {
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: #6E6E73 !important;
}

.score-display input {
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    color: #1D1D1F !important;
    text-align: center !important;
    border: none !important;
    background: transparent !important;
}

/* Image display */
.molecule-image {
    border-radius: 12px !important;
    overflow: hidden !important;
    background: #FFFFFF !important;
}

/* Target molecule preview */
.target-preview-container {
    position: relative;
}

.target-preview {
    position: absolute !important;
    top: 8px !important;
    right: 8px !important;
    width: 120px !important;
    height: 120px !important;
    border-radius: 8px !important;
    border: 1px solid #E8E8ED !important;
    background: #FFFFFF !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
    z-index: 10 !important;
    overflow: hidden !important;
}

.target-preview img {
    width: 100% !important;
    height: 100% !important;
    object-fit: contain !important;
}

.target-preview-label {
    position: absolute !important;
    bottom: 0 !important;
    left: 0 !important;
    right: 0 !important;
    background: rgba(0, 0, 0, 0.6) !important;
    color: #FFFFFF !important;
    font-size: 0.65rem !important;
    text-align: center !important;
    padding: 2px 4px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* Molecule display wrapper */
.molecule-display-wrapper {
    position: relative !important;
}

/* Target thumbnail */
.target-thumbnail {
    border-radius: 8px !important;
    border: 1px solid #E8E8ED !important;
    background: #FFFFFF !important;
    max-width: 100px !important;
}

.target-thumbnail img {
    object-fit: contain !important;
}

/* Fix double borders on slider number inputs */
input[type="number"] {
    border: 1px solid #D2D2D7 !important;
    box-shadow: none !important;
    outline: none !important;
}

input[type="number"]:focus {
    border-color: #007AFF !important;
    box-shadow: none !important;
}

/* Remove wrapper borders around number inputs in sliders */
[data-testid="number-input"],
.number-input,
.gradio-slider .wrap,
.gradio-slider > div > div {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}

/* Summary text */
.summary-text {
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
    color: #1D1D1F !important;
}

/* Judge result */
.judge-result {
    background: #F0FDF4 !important;
    border: 1px solid #BBF7D0 !important;
    border-radius: 10px !important;
    padding: 16px !important;
    margin-top: 16px !important;
}

.judge-result.warning {
    background: #FFFBEB !important;
    border-color: #FDE68A !important;
}

/* Feedback section */
.feedback-section {
    background: #FFFFFF !important;
    border-radius: 16px !important;
    padding: 24px !important;
    margin-top: 16px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
}

/* Constraints list */
.constraints-list {
    background: #F5F5F7 !important;
    border-radius: 10px !important;
    padding: 16px !important;
    margin-top: 16px !important;
}

/* Navigation display */
.nav-display input {
    text-align: center !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}

/* Status messages */
.status-message {
    padding: 12px 16px !important;
    border-radius: 10px !important;
    font-size: 0.9rem !important;
}

/* Dropdown */
.gradio-container .gradio-dropdown {
    border-radius: 10px !important;
}

/* Hide default borders on groups */
.gradio-group {
    border: none !important;
    background: transparent !important;
}

/* Row spacing */
.gradio-row {
    gap: 16px !important;
}

/* Column spacing */
.gradio-column {
    gap: 12px !important;
}
"""


def create_app() -> tuple[gr.Blocks, gr.themes.Base, str]:
    """Create the Gradio application.
    
    Returns:
        Tuple of (app, theme, css) for Gradio 6.0+ compatibility.
    """
    # Create custom theme
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.gray,
        neutral_hue=gr.themes.colors.gray,
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        # Colors
        body_background_fill="#F5F5F7",
        block_background_fill="#FFFFFF",
        block_border_width="0px",
        block_shadow="0 1px 3px rgba(0, 0, 0, 0.08), 0 4px 12px rgba(0, 0, 0, 0.04)",
        block_radius="16px",
        
        # Buttons
        button_primary_background_fill="#007AFF",
        button_primary_background_fill_hover="#0066D6",
        button_primary_text_color="#FFFFFF",
        button_primary_border_color="transparent",
        button_secondary_background_fill="#FFFFFF",
        button_secondary_background_fill_hover="#F5F5F7",
        button_secondary_text_color="#1D1D1F",
        button_secondary_border_color="#D2D2D7",
        
        # Inputs
        input_background_fill="#FFFFFF",
        input_border_color="#D2D2D7",
        input_border_color_focus="#007AFF",
        input_radius="10px",
        
        # Text
        body_text_color="#1D1D1F",
        body_text_color_subdued="#6E6E73",
        
        # Spacing
        block_padding="24px",
        layout_gap="16px",
    )
    
    with gr.Blocks(
        title="Molecule Optimization",
    ) as app:
        # Session state
        session_state = gr.State(None)
        
        # Header
        gr.HTML("""
            <div class="main-header">
                <h1>Molecule Optimization</h1>
                <p>AI-powered molecular optimization for similarity and drug-likeness</p>
            </div>
        """)
        
        with gr.Row():
            # Left column: Configuration
            with gr.Column(scale=1):
                gr.HTML('<div class="card-header">Configuration</div>')
                
                task_dropdown = gr.Dropdown(
                    choices=["similarity_qed"],
                    value="similarity_qed",
                    label="Task",
                    info="Select optimization objective",
                    interactive=True,
                )
                
                target_smiles = gr.Textbox(
                    label="Target SMILES",
                    value=DEFAULT_TARGET_SMILES,
                    info="Reference molecule for similarity calculation",
                    lines=2,
                    placeholder="Enter SMILES string...",
                )
                
                target_score = gr.Slider(
                    minimum=0.5,
                    maximum=1.0,
                    value=0.75,
                    step=0.01,
                    label="Target Score",
                    info="Combined score threshold",
                )
                
                with gr.Row():
                    min_similarity = gr.Slider(
                        minimum=0.3,
                        maximum=1.0,
                        value=0.7,
                        step=0.01,
                        label="Min Similarity",
                        info="MACCS similarity",
                    )
                    min_qed = gr.Slider(
                        minimum=0.3,
                        maximum=1.0,
                        value=0.7,
                        step=0.01,
                        label="Min QED",
                        info="Drug-likeness score",
                    )
                
                start_btn = gr.Button(
                    "Start Optimization", 
                    variant="primary", 
                    size="lg",
                    elem_classes=["primary-btn"],
                )
            
            # Right column: Results
            with gr.Column(scale=2):
                gr.HTML('<div class="card-header">Result</div>')
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Molecule display with target preview
                        result_image = gr.Image(
                            label="Structure",
                            type="pil",
                            height=320,
                            elem_classes=["molecule-image"],
                            show_label=False,
                        )
                        # Target molecule preview
                        gr.HTML('<span style="font-size: 0.75rem; color: #6E6E73; display: block; margin-bottom: 4px;">Target molecule</span>')
                        target_image = gr.Image(
                            type="pil",
                            height=100,
                            show_label=False,
                            interactive=False,
                            elem_classes=["target-thumbnail"],
                        )
                        
                        result_smiles = gr.Textbox(
                            label="SMILES",
                            interactive=False,
                        )
                        
                        with gr.Row():
                            score_similarity = gr.Number(
                                label="Similarity", 
                                interactive=False,
                                elem_classes=["score-display"],
                            )
                            score_qed = gr.Number(
                                label="QED", 
                                interactive=False,
                                elem_classes=["score-display"],
                            )
                            score_combined = gr.Number(
                                label="Combined", 
                                interactive=False,
                                elem_classes=["score-display"],
                            )
                        
                        # Navigation controls
                        with gr.Row():
                            prev_btn = gr.Button(
                                "← Previous", 
                                size="sm", 
                                scale=1,
                                elem_classes=["nav-btn"],
                            )
                            nav_display = gr.Textbox(
                                value="0 / 0",
                                label="",
                                interactive=False,
                                scale=1,
                                elem_classes=["nav-display"],
                                show_label=False,
                            )
                            next_btn = gr.Button(
                                "Next →", 
                                size="sm", 
                                scale=1,
                                elem_classes=["nav-btn"],
                            )
                        
                        iterations_display = gr.Number(
                            label="Total Iterations", 
                            interactive=False,
                        )
                    
                    with gr.Column(scale=1):
                        gr.HTML('<div class="card-header" style="border:none; padding-bottom:0;">Summary</div>')
                        
                        summary_text = gr.Markdown(
                            value="Run optimization to see results.",
                            elem_classes=["summary-text"],
                        )
                        
                        judge_result_text = gr.Markdown(
                            value="",
                            visible=False,
                            elem_classes=["judge-result"],
                        )
                        
                        # Save button
                        save_btn = gr.Button(
                            "Save Conversation", 
                            variant="secondary", 
                            visible=False,
                            elem_classes=["secondary-btn"],
                        )
                        save_status = gr.Markdown(value="", visible=False)
        
        # Feedback section
        with gr.Row(visible=False) as feedback_section:
            with gr.Column(elem_classes=["feedback-section"]):
                gr.HTML('<div class="card-header">Feedback</div>')
                gr.Markdown(
                    "Not satisfied with the result? Describe what changes you'd like. "
                    "Your feedback accumulates across optimization rounds.",
                    elem_classes=["summary-text"],
                )
                
                feedback_input = gr.Textbox(
                    label="Your Feedback",
                    placeholder="e.g., 'Remove the carbonic acid group' or 'Make it more rigid'",
                    lines=2,
                )
                
                with gr.Row():
                    continue_btn = gr.Button(
                        "Continue Optimization", 
                        variant="primary",
                        elem_classes=["primary-btn"],
                    )
                    reset_btn = gr.Button(
                        "Reset Session", 
                        variant="secondary",
                        elem_classes=["secondary-btn"],
                    )
                
                accumulated_constraints = gr.Markdown(
                    value="**Accumulated Constraints:** None yet",
                    elem_classes=["constraints-list"],
                )
        
        # Status display
        status_text = gr.Markdown(value="", visible=False)
        
        # --- Event Handlers ---
        
        def update_target_image(smiles: str):
            """Update target molecule preview image."""
            try:
                return smiles_to_image(smiles, size=(150, 150))
            except Exception:
                return None
        
        def start_optimization(
            target_smiles_val: str,
            target_score_val: float,
            min_similarity_val: float,
            min_qed_val: float,
        ):
            """Start a new optimization session with streaming updates."""
            
            # Create new session
            session = InteractiveSession(
                target_smiles=target_smiles_val,
                target_score=target_score_val,
                min_similarity=min_similarity_val,
                min_qed=min_qed_val,
            )
            
            # Stream through optimization
            for item in session.start_streaming():
                if isinstance(item, TraceEntry):
                    # Intermediate update - show current molecule
                    nav_text = f"{item.iteration} / ..."
                    
                    if item.is_valid:
                        yield (
                            session,  # session_state
                            item.image,  # result_image
                            item.smiles,  # result_smiles
                            round(item.scores.get("Similarity", 0.0), 2) if item.scores else None,
                            round(item.scores.get("QED", 0.0), 2) if item.scores else None,
                            round(item.combined_score, 2) if item.combined_score else None,
                            nav_text,  # nav_display
                            item.iteration,  # iterations_display
                            f"Optimizing — Iteration {item.iteration}",  # summary_text
                            "",  # judge_result_text
                            gr.update(visible=False),  # judge_result visibility
                            gr.update(visible=False),  # feedback_section
                            gr.update(visible=False),  # save_btn
                            gr.update(visible=False),  # save_status
                            "**Accumulated Constraints:** None yet",  # accumulated_constraints
                            "",  # feedback_input
                            gr.update(visible=False),  # status_text
                        )
                    else:
                        # Invalid SMILES
                        yield (
                            session,
                            None,  # No image
                            f"[Invalid] {item.smiles}",
                            None, None, None,
                            nav_text,
                            item.iteration,
                            f"Optimizing — Iteration {item.iteration} (validating...)",
                            "",
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            "**Accumulated Constraints:** None yet",
                            "",
                            gr.update(visible=False),
                        )
                
                elif isinstance(item, SessionResult):
                    # Final result
                    nav_text = f"{session.current_trace_index + 1} / {session.get_trace_length()}"
                    
                    yield (
                        session,
                        item.image,
                        item.smiles,
                        round(item.scores.get("Similarity", 0.0), 2),
                        round(item.scores.get("QED", 0.0), 2),
                        round(item.combined_score, 2),
                        nav_text,
                        item.iteration_count,
                        item.summary,
                        "",
                        gr.update(visible=False),
                        gr.update(visible=True),  # Show feedback section
                        gr.update(visible=True),  # Show save button
                        gr.update(visible=False),
                        "**Accumulated Constraints:** None yet",
                        "",
                        gr.update(visible=False),
                    )
        
        def continue_optimization(
            session: InteractiveSession,
            feedback: str,
        ):
            """Continue optimization with user feedback."""
            
            if session is None:
                yield (
                    None,
                    None, "", None, None, None, "0 / 0", 0,
                    "No active session. Please start optimization first.",
                    "", gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    "**Accumulated Constraints:** None",
                    "",
                    gr.update(visible=True, value="No active session"),
                )
                return
            
            if not feedback.strip():
                yield (
                    session,
                    None, "", None, None, None, "0 / 0", 0,
                    "Please provide feedback to continue.",
                    "", gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    _format_constraints(session.get_accumulated_constraints()),
                    feedback,
                    gr.update(visible=True, value="Please enter feedback"),
                )
                return
            
            # Stream through optimization
            for item in session.continue_streaming(feedback.strip()):
                if isinstance(item, TraceEntry):
                    nav_text = f"{item.iteration} / ..."
                    
                    if item.is_valid:
                        yield (
                            session,
                            item.image,
                            item.smiles,
                            round(item.scores.get("Similarity", 0.0), 2) if item.scores else None,
                            round(item.scores.get("QED", 0.0), 2) if item.scores else None,
                            round(item.combined_score, 2) if item.combined_score else None,
                            nav_text,
                            item.iteration,
                            f"Optimizing — Iteration {item.iteration}",
                            "",
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            _format_constraints(session.get_accumulated_constraints()),
                            "",
                            gr.update(visible=False),
                        )
                    else:
                        yield (
                            session,
                            None,
                            f"[Invalid] {item.smiles}",
                            None, None, None,
                            nav_text,
                            item.iteration,
                            f"Optimizing — Iteration {item.iteration} (validating...)",
                            "",
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            _format_constraints(session.get_accumulated_constraints()),
                            "",
                            gr.update(visible=False),
                        )
                
                elif isinstance(item, SessionResult):
                    nav_text = f"{session.current_trace_index + 1} / {session.get_trace_length()}"
                    
                    judge_text = ""
                    if item.judge_result:
                        status_text = "Constraints satisfied" if item.judge_result.satisfied else "Constraints not fully satisfied"
                        judge_text = f"""**Evaluation:** {status_text}

{item.judge_result.reason}"""
                    
                    yield (
                        session,
                        item.image,
                        item.smiles,
                        round(item.scores.get("Similarity", 0.0), 2),
                        round(item.scores.get("QED", 0.0), 2),
                        round(item.combined_score, 2),
                        nav_text,
                        item.iteration_count,
                        item.summary,
                        judge_text,
                        gr.update(visible=bool(judge_text)),
                        gr.update(visible=True),
                        gr.update(visible=True),
                        gr.update(visible=False),
                        _format_constraints(session.get_accumulated_constraints()),
                        "",
                        gr.update(visible=False),
                    )
        
        def navigate_previous(session: InteractiveSession):
            """Navigate to previous molecule in trace."""
            if session is None or session.get_trace_length() == 0:
                return (
                    session, None, "", None, None, None, "0 / 0"
                )
            
            entry = session.navigate_previous()
            if entry is None:
                entry = session.get_current_entry()
            
            if entry is None:
                return (session, None, "", None, None, None, "0 / 0")
            
            nav_text = f"{session.current_trace_index + 1} / {session.get_trace_length()}"
            
            if entry.is_valid:
                return (
                    session,
                    entry.image,
                    entry.smiles,
                    round(entry.scores.get("Similarity", 0.0), 2) if entry.scores else None,
                    round(entry.scores.get("QED", 0.0), 2) if entry.scores else None,
                    round(entry.combined_score, 2) if entry.combined_score else None,
                    nav_text,
                )
            else:
                return (
                    session,
                    None,
                    f"[INVALID] {entry.smiles}",
                    None, None, None,
                    nav_text,
                )
        
        def navigate_next(session: InteractiveSession):
            """Navigate to next molecule in trace."""
            if session is None or session.get_trace_length() == 0:
                return (
                    session, None, "", None, None, None, "0 / 0"
                )
            
            entry = session.navigate_next()
            if entry is None:
                entry = session.get_current_entry()
            
            if entry is None:
                return (session, None, "", None, None, None, "0 / 0")
            
            nav_text = f"{session.current_trace_index + 1} / {session.get_trace_length()}"
            
            if entry.is_valid:
                return (
                    session,
                    entry.image,
                    entry.smiles,
                    round(entry.scores.get("Similarity", 0.0), 2) if entry.scores else None,
                    round(entry.scores.get("QED", 0.0), 2) if entry.scores else None,
                    round(entry.combined_score, 2) if entry.combined_score else None,
                    nav_text,
                )
            else:
                return (
                    session,
                    None,
                    f"[INVALID] {entry.smiles}",
                    None, None, None,
                    nav_text,
                )
        
        def save_conversation(session: InteractiveSession):
            """Save the conversation to a file."""
            if session is None:
                return gr.update(visible=True, value="No session to save")
            
            try:
                filepath = session.save_conversation()
                return gr.update(visible=True, value=f"Saved to: `{filepath}`")
            except Exception as e:
                return gr.update(visible=True, value=f"Error saving: {str(e)}")
        
        def reset_session():
            """Reset the session."""
            return (
                None,  # session_state
                None,  # result_image
                "",  # result_smiles
                None,  # score_similarity
                None,  # score_qed
                None,  # score_combined
                "0 / 0",  # nav_display
                0,  # iterations_display
                "Run optimization to see results.",  # summary_text
                "",  # judge_result_text
                gr.update(visible=False),  # judge_result visibility
                gr.update(visible=False),  # feedback_section
                gr.update(visible=False),  # save_btn
                gr.update(visible=False),  # save_status
                "**Accumulated Constraints:** None yet",  # accumulated_constraints
                "",  # feedback_input
                gr.update(visible=False),  # status_text
            )
        
        def _format_constraints(constraints: list[str]) -> str:
            """Format accumulated constraints for display."""
            if not constraints:
                return "**Accumulated Constraints:** None yet"
            
            items = "\n".join(f"- {c}" for c in constraints)
            return f"**Accumulated Constraints:**\n{items}"
        
        # Define outputs for optimization functions
        optimization_outputs = [
            session_state,
            result_image,
            result_smiles,
            score_similarity,
            score_qed,
            score_combined,
            nav_display,
            iterations_display,
            summary_text,
            judge_result_text,
            judge_result_text,  # visibility update
            feedback_section,
            save_btn,
            save_status,
            accumulated_constraints,
            feedback_input,
            status_text,
        ]
        
        navigation_outputs = [
            session_state,
            result_image,
            result_smiles,
            score_similarity,
            score_qed,
            score_combined,
            nav_display,
        ]
        
        # Wire up events
        start_btn.click(
            fn=start_optimization,
            inputs=[target_smiles, target_score, min_similarity, min_qed],
            outputs=optimization_outputs,
        )
        
        continue_btn.click(
            fn=continue_optimization,
            inputs=[session_state, feedback_input],
            outputs=optimization_outputs,
        )
        
        prev_btn.click(
            fn=navigate_previous,
            inputs=[session_state],
            outputs=navigation_outputs,
        )
        
        next_btn.click(
            fn=navigate_next,
            inputs=[session_state],
            outputs=navigation_outputs,
        )
        
        save_btn.click(
            fn=save_conversation,
            inputs=[session_state],
            outputs=[save_status],
        )
        
        reset_btn.click(
            fn=reset_session,
            outputs=optimization_outputs,
        )
        
        # Update target image when SMILES changes
        target_smiles.change(
            fn=update_target_image,
            inputs=[target_smiles],
            outputs=[target_image],
        )
        
        # Initialize target image on load
        app.load(
            fn=update_target_image,
            inputs=[target_smiles],
            outputs=[target_image],
        )
    
    return app, theme, CUSTOM_CSS


def main():
    """Main entry point for the UI."""
    load_dotenv()
    
    app, theme, css = create_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=theme,
        css=css,
    )
    return app


if __name__ == "__main__":
    main()
