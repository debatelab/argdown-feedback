"""
Test for the Argdown Analysis OpenEnv environment.

This test demonstrates the basic usage of the ArgdownAnalysisEnv client
with Docker image deployment and basic reset/step operations.

Note: First test run may be slow due to dataset downloads from HuggingFace Hub.
For faster tests in development/CI, consider:
1. Pre-building image with cached datasets
2. Using a shared volume mount for HF_DATASETS_CACHE
3. Running tests against a long-running container instead of from_docker_image()
"""

from envs.argdown_analysis import ArgdownAnalysisAction, ArgdownAnalysisEnv  # type: ignore[import]


def test_argdown_analysis_env_basic():
    """Test basic reset and step operations with the ArgdownAnalysisEnv."""
    
    # Automatically start container and connect
    client = ArgdownAnalysisEnv.from_docker_image("ghcr.io/debatelab/openenv-argdown-analysis:latest")
    
    try:
        # Reset the environment (no parameters - samples automatically)
        result = client.reset()
        
        # Verify we got a prompt
        assert result.observation.prompt is not None
        assert len(result.observation.prompt) > 0
        print(f"Initial prompt: {result.observation.prompt[:100]}...")
        
        # Verify state has been populated
        state = client.state()
        assert state.task_id is not None
        assert state.source_text is not None
        assert len(state.source_text) > 0
        assert state.subtask_id is not None
        print(f"Sampled task: {state.task_id}")
        print(f"Source text length: {len(state.source_text)} chars")
        
        # Send a message with an argument reconstruction
        action = ArgdownAnalysisAction(
            message="<think>Starting with informal argument reco.</think>\n"
                   "```argdown\n"
                   "<Arg1>: Democracy is beneficial\n"
                   "  (1) Democracy promotes freedom.\n"
                   "  (2) Freedom is valuable.\n"
                   "  ----\n"
                   "  (3) Therefore, democracy is beneficial.\n"
                   "```"
        )
        
        result = client.step(action)
        
        # Verify response
        assert result.observation.prompt is not None

        print(f"Response prompt: {result.observation.prompt[:100]}...")
        print(f"Reward: {result.reward}")
        print(f"Done: {result.done}")
        
        # Check that we got some reward value (can be 0 or positive)
        assert result.reward is not None
        assert result.reward >= 0
        
        # The task should not be done after just one step
        assert isinstance(result.done, bool)
        
    finally:
        # Cleanup - stops and removes container
        # Note: First run may timeout during cleanup due to dataset download
        try:
            client.close()
        except Exception as e:
            print(f"Warning: Container cleanup encountered an issue: {type(e).__name__}")
            # This is often due to slow dataset download on first run
            # The container will be cleaned up eventually by Docker


def test_argdown_analysis_env_state():
    """Test state retrieval from the ArgdownAnalysisEnv."""
    
    client = ArgdownAnalysisEnv.from_docker_image("ghcr.io/debatelab/openenv-argdown-analysis:latest")
    
    try:
        # Reset the environment (automatic sampling)
        client.reset()
        
        # Get state
        state = client.state()
        
        # Verify state properties exist
        assert hasattr(state, 'episode_id')
        assert hasattr(state, 'step_count')
        assert hasattr(state, 'task_id')
        assert hasattr(state, 'source_text')
        assert hasattr(state, 'subtask_id')
        assert hasattr(state, 'subtasks_completed')
        assert hasattr(state, 'history')
        
        # Verify initial state values
        assert state.step_count == 0
        assert state.task_id is not None
        assert len(state.source_text) > 0
        assert state.subtask_id is not None
        assert len(state.subtasks_completed) == 0
        
        # Take a step
        action = ArgdownAnalysisAction(
            message="```argdown\n<Arg>: Climate argument\n```"
        )
        client.step(action)
        
        # Get updated state
        state = client.state()
        assert state.step_count == 1
        # History contains entries for initial prompt, user message, and next prompt
        assert len(state.history) >= 2
        
    finally:
        try:
            client.close()
        except Exception as e:
            print(f"Warning: Container cleanup encountered an issue: {type(e).__name__}")


def test_argdown_analysis_env_multi_step():
    """Test multiple steps in the ArgdownAnalysisEnv."""
    
    client = ArgdownAnalysisEnv.from_docker_image("ghcr.io/debatelab/openenv-argdown-analysis:latest")
    
    try:
        # Reset (automatic sampling from configured datasets)
        client.reset()
        
        # Take multiple steps
        for i in range(3):
            action = ArgdownAnalysisAction(
                message=f"<think>Step {i+1}</think>\n```argdown\n<Arg{i+1}>: Test\n```"
            )
            result = client.step(action)
            
            assert result.observation.prompt is not None
            print(f"Step {i+1} - Reward: {result.reward}, Done: {result.done}")
            
            # If done, break
            if result.done:
                print(f"Task completed after {i+1} steps")
                break
        
        # Verify state
        state = client.state()
        assert state.step_count >= 1
        assert len(state.history) >= 1
        
    finally:
        try:
            client.close()
        except Exception as e:
            print(f"Warning: Container cleanup encountered an issue: {type(e).__name__}")


def test_argdown_analysis_env_config_based():
    """Test that environment respects configuration and samples from datasets."""
    
    client = ArgdownAnalysisEnv.from_docker_image("ghcr.io/debatelab/openenv-argdown-analysis:latest")
    
    try:
        # Reset multiple times and verify we get valid sampled data
        sampled_tasks = set()
        
        for i in range(5):
            result = client.reset()
            state = client.state()
            
            # Verify task was sampled from valid options
            assert state.task_id is not None
            task_name = state.task_id.value if hasattr(state.task_id, 'value') else str(state.task_id)
            assert task_name in ["SingleArgumentAnalysis", "MultiArgumentAnalysis"]
            sampled_tasks.add(task_name)
            
            # Verify source text is from dataset (not empty, reasonable length)
            assert state.source_text is not None
            assert len(state.source_text) > 10, "Source text too short - likely not from dataset"
            assert len(state.source_text) < 50000, "Source text unreasonably long"
            
            # Verify subtask is set
            assert state.subtask_id is not None
            assert len(state.subtask_id) > 0
            
            # Verify initial prompt
            assert result.observation.prompt is not None
            assert len(result.observation.prompt) > 0
            
            print(f"Reset {i+1}: task={task_name}, source_len={len(state.source_text)}, subtask={state.subtask_id}")
        
        print(f"\nSampled tasks across 5 resets: {sampled_tasks}")
        
    finally:
        try:
            client.close()
        except Exception as e:
            print(f"Warning: Container cleanup encountered an issue: {type(e).__name__}")


if __name__ == "__main__":
    # Run tests directly
    print("Running test_argdown_analysis_env_basic...")
    test_argdown_analysis_env_basic()
    print("\n" + "="*50 + "\n")
    
    print("Running test_argdown_analysis_env_state...")
    test_argdown_analysis_env_state()
    print("\n" + "="*50 + "\n")
    
    print("Running test_argdown_analysis_env_multi_step...")
    test_argdown_analysis_env_multi_step()
    print("\n" + "="*50 + "\n")
    
    print("Running test_argdown_analysis_env_config_based...")
    test_argdown_analysis_env_config_based()
    print("\nAll tests completed!")
