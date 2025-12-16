"""Gradio UI for interactive molecule optimization."""

import gradio as gr
from dotenv import load_dotenv

from .runner import InteractiveSession, SessionResult, TraceEntry, DEFAULT_TARGET_SMILES


def create_app() -> gr.Blocks:
    """Create the Gradio application."""
    
    with gr.Blocks(
        title="Molecule Optimization Agent",
    ) as app:
        # Session state
        session_state = gr.State(None)
        
        gr.Markdown("""
        # 🧬 Molecule Optimization Agent
        
        Optimize molecules for similarity and drug-likeness using an LLM-driven iterative approach.
        Provide feedback to refine the results!
        """)
        
        with gr.Row():
            # Left column: Configuration
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Configuration")
                
                task_dropdown = gr.Dropdown(
                    choices=["similarity_qed"],
                    value="similarity_qed",
                    label="Task",
                    info="Optimization task (more coming soon)",
                    interactive=True,
                )
                
                target_smiles = gr.Textbox(
                    label="Target SMILES",
                    value=DEFAULT_TARGET_SMILES,
                    info="Reference molecule for similarity (default: Quercetin)",
                    lines=2,
                )
                
                with gr.Row():
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
                        info="Minimum MACCS fingerprint similarity",
                    )
                    min_qed = gr.Slider(
                        minimum=0.3,
                        maximum=1.0,
                        value=0.7,
                        step=0.01,
                        label="Min QED",
                        info="Minimum drug-likeness score",
                    )
                
                start_btn = gr.Button("▶️ Start Optimization", variant="primary", size="lg")
        
        gr.Markdown("---")
        
        # Results section
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 Result")
                
                result_image = gr.Image(
                    label="Molecule Structure",
                    type="pil",
                    height=400,
                )
                
                result_smiles = gr.Textbox(
                    label="SMILES",
                    interactive=False,
                )
                
                with gr.Row():
                    score_similarity = gr.Number(label="Similarity", interactive=False)
                    score_qed = gr.Number(label="QED", interactive=False)
                    score_combined = gr.Number(label="Combined", interactive=False)
                
                # Navigation controls
                with gr.Row():
                    prev_btn = gr.Button("⬅️ Previous", size="sm", scale=1)
                    nav_display = gr.Textbox(
                        value="0 / 0",
                        label="Molecule",
                        interactive=False,
                        scale=1,
                    )
                    next_btn = gr.Button("Next ➡️", size="sm", scale=1)
                
                iterations_display = gr.Number(label="Total Iterations", interactive=False)
            
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Summary")
                
                summary_text = gr.Markdown(
                    value="*Run optimization to see results...*",
                )
                
                judge_result_text = gr.Markdown(
                    value="",
                    visible=False,
                )
                
                # Save button
                save_btn = gr.Button("💾 Save Conversation", variant="secondary", visible=False)
                save_status = gr.Markdown(value="", visible=False)
        
        gr.Markdown("---")
        
        # Feedback section
        with gr.Row(visible=False) as feedback_section:
            with gr.Column():
                gr.Markdown("### 💬 Provide Feedback")
                gr.Markdown(
                    "*Not satisfied? Tell the agent what to change. "
                    "Your feedback will be accumulated across rounds.*"
                )
                
                feedback_input = gr.Textbox(
                    label="Your Feedback",
                    placeholder="e.g., 'Remove the carbonic acid group' or 'Make it less bulky'",
                    lines=2,
                )
                
                with gr.Row():
                    continue_btn = gr.Button("🔄 Continue Optimization", variant="primary")
                    reset_btn = gr.Button("🗑️ Reset Session", variant="secondary")
                
                accumulated_constraints = gr.Markdown(
                    value="**Accumulated Constraints:** None yet",
                )
        
        # Status display
        status_text = gr.Markdown(value="", visible=False)
        
        # --- Event Handlers ---
        
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
                            f"*Optimizing... Iteration {item.iteration}*",  # summary_text
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
                            f"[INVALID] {item.smiles}",
                            None, None, None,
                            nav_text,
                            item.iteration,
                            f"*Optimizing... Iteration {item.iteration} (invalid SMILES)*",
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
                    "*Error: No active session. Start optimization first.*",
                    "", gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    "**Accumulated Constraints:** None",
                    "",
                    gr.update(visible=True, value="⚠️ Error: No active session"),
                )
                return
            
            if not feedback.strip():
                yield (
                    session,
                    None, "", None, None, None, "0 / 0", 0,
                    "*Please provide feedback to continue.*",
                    "", gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    _format_constraints(session.get_accumulated_constraints()),
                    feedback,
                    gr.update(visible=True, value="⚠️ Please enter feedback"),
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
                            f"*Optimizing... Iteration {item.iteration}*",
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
                            f"[INVALID] {item.smiles}",
                            None, None, None,
                            nav_text,
                            item.iteration,
                            f"*Optimizing... Iteration {item.iteration} (invalid SMILES)*",
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
                        status_emoji = "✅" if item.judge_result.satisfied else "⚠️"
                        judge_text = f"""
**Judge Evaluation:** {status_emoji} {"Satisfied" if item.judge_result.satisfied else "Not fully satisfied"}

*{item.judge_result.reason}*
"""
                    
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
                return gr.update(visible=True, value="⚠️ No session to save")
            
            try:
                filepath = session.save_conversation()
                return gr.update(visible=True, value=f"✅ Saved to: `{filepath}`")
            except Exception as e:
                return gr.update(visible=True, value=f"⚠️ Error saving: {str(e)}")
        
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
                "*Run optimization to see results...*",  # summary_text
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
    
    return app


def main():
    """Main entry point for the UI."""
    load_dotenv()
    
    app = create_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )
    return app


if __name__ == "__main__":
    main()
