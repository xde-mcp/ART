import art
import asyncio
from dotenv import load_dotenv
import json
import openai
import random
import re
from typing import TypedDict
from typing import List, Tuple, Literal
from art.utils.get_trajectory_messages import get_trajectory_messages

load_dotenv()

class ConnectionPuzzle(TypedDict):
    words: List[str]
    solution: List[Tuple[Literal["yellow", "green", "blue", "purple"], str, List[str]]]


def load_puzzles(file_path: str) -> List[ConnectionPuzzle]:
    """
    Load puzzles from a JSONL file and convert them to ConnectionPuzzle format.
    
    Args:
        file_path: Path to the JSONL file containing puzzle data
        
    Returns:
        List of ConnectionPuzzle objects
    """
    puzzles = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
                
            data = json.loads(line)
            assistant_content = json.loads(data["messages"][1]["content"])
            
            # Extract categories from the puzzle
            categories = assistant_content.get("categories", [])
            
            # Create a flat list of all words
            all_words = []
            solution = []
            
            for category in categories:
                color = category["color"]
                category_name = category["category"]
                words = category["words"]
                
                all_words.extend(words)
                solution.append((color, category_name, words))
            
            # Randomize the order of words
            random.shuffle(all_words)
            
            # Create the ConnectionPuzzle
            puzzle = ConnectionPuzzle(
                words=all_words,
                solution=solution
            )
            
            puzzles.append(puzzle)
    
    return puzzles


async def rollout(
    client: openai.AsyncOpenAI, puzzle: ConnectionPuzzle
) -> art.Trajectory:
    
    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content":
                """
                You are an expert at solving the New York Times Connection puzzle game.

                In Connection, you are given 16 words and need to group them into 4 categories of 4 words each.
                Each category has a theme that connects its 4 words. Categories are color-coded by difficulty:
                - Yellow (easiest): Most straightforward connections
                - Green (easy): Slightly more subtle connections
                - Blue (medium): More challenging connections
                - Purple (hardest): Most difficult, often requiring lateral thinking

                When solving, analyze the words carefully and look for potential relationships between them.
                Consider different meanings, contexts, and patterns. Words might be related by:
                - Belonging to the same category or field
                - Being synonyms or related concepts
                - Forming common phrases or expressions
                - Sharing word patterns or linguistic features

                Your task is to identify all 4 categories and their 4 words each. 
                
                First, provide your reasoning within <reasoning></reasoning> tags, explaining how you identified each category and why the words belong together.
                
                Then provide your solution in JSON format within <output></output> tags as follows:

                <reasoning>
                Explain your thought process for identifying each category and why the words belong together.
                </reasoning>

                <output>
                [
                  {
                    "color": "yellow",
                    "category": "category name",
                    "words": ["word1", "word2", "word3", "word4"]
                  },
                  {
                    "color": "green",
                    "category": "category name",
                    "words": ["word1", "word2", "word3", "word4"]
                  },
                  {
                    "color": "blue",
                    "category": "category name",
                    "words": ["word1", "word2", "word3", "word4"]
                  },
                  {
                    "color": "purple",
                    "category": "category name",
                    "words": ["word1", "word2", "word3", "word4"]
                  }
                ]
                </output>

                Be precise with your category names, as they should clearly explain the connection between the words.
                """,
            }
        ],
        reward=0,
    )
    trajectory.messages_and_choices.append(
        {
            "role": "user",
            "content": puzzle["words"]
        }
    )

    messages = get_trajectory_messages(trajectory)

    chat_completion = await client.chat.completions.create(
        messages=messages, model=model.name
    )
    choice = chat_completion.choices[0]
    trajectory.messages_and_choices.append(choice)

    content = choice.message.content
    
    # Extract JSON output from the response
    output_match = re.search(r'<output>(.*?)</output>', content, re.DOTALL)
    if not output_match:
        return trajectory
    
    try:
        model_solution = json.loads(output_match.group(1).strip())
        
        # Initialize counters for correct categories
        weighted_correct = 0
        color_weights = {"yellow": 1, "green": 1.25, "blue": 1.5, "purple": 2}
        
        # Check each category in the solution
        for color, category_name, expected_words in puzzle["solution"]:
            # Find matching category in model solution
            for model_category in model_solution:
                if model_category["color"] == color:
                    # Check if words match (order-independent)
                    model_words = set(word.lower() for word in model_category["words"])
                    expected_words_set = set(word.lower() for word in expected_words)
                    
                    # Calculate partial correctness - how many words match
                    correct_words = len(model_words.intersection(expected_words_set))
                    word_accuracy = correct_words / 4 
                    weighted_word_accuracy = color_weights[color] * word_accuracy

                    weighted_correct += weighted_word_accuracy
                    
                    # Debug print statement
                    print(f"Category {color}: {correct_words}/4 words correct, accuracy: {word_accuracy:.2f}, weighted: {weighted_word_accuracy:.2f}")
                    
                    break
        
        # Calculate metrics
        max_weighted_score = sum(color_weights.values())
        weighted_accuracy = weighted_correct / max_weighted_score
        
        trajectory.reward = weighted_accuracy

        return trajectory
    except (json.JSONDecodeError, KeyError, TypeError):
        return trajectory
    


async def main():
    connection_puzzles: list[ConnectionPuzzle] = load_puzzles("examples/connection/puzzle.jsonl")
    print(connection_puzzles[0])
    val_puzzles = connection_puzzles[:90]
    test_puzzles = connection_puzzles[90:140]
    train_puzzles = connection_puzzles[140:]
    random.seed(42)
    random.shuffle(train_puzzles)

    model = art.TrainableModel(
        name="001",
        project="connection",
        base_model="Qwen/Qwen2.5-14B-Instruct",
        _internal_config={"init_args": {"gpu_memory_utilization": 0.775}},
    )
    await model.register(art.LocalAPI())

    batch_size = 4  # Previously called stride
    num_epochs = 1  # Number of complete passes through the training data
    openai_client = model.openai_client()
    
    start_step = await model.get_step()
    max_steps = 1000
    
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}")
        # Shuffle training data at the beginning of each epoch
        random.shuffle(train_puzzles)
        
        # Calculate how many batches we can process in this epoch
        num_batches = min(len(train_puzzles) // batch_size, (max_steps - start_step) // num_epochs)
        
        for batch in range(num_batches):
            current_step = start_step + epoch * num_batches + batch
            if current_step >= max_steps:
                break
                
            print(f"Epoch {epoch+1}, Batch {batch+1}/{num_batches}, Step {current_step}")
            
            batch_start_idx = batch * batch_size
            batch_end_idx = (batch + 1) * batch_size
            
            val_groups, train_groups = await asyncio.gather(
                art.gather_trajectory_groups(
                    (
                        art.TrajectoryGroup(rollout(openai_client, puzzle) for _ in range(2))
                        for puzzle in val_puzzles
                    ),
                    pbar_desc=f"val (epoch {epoch+1})",
                ),
                art.gather_trajectory_groups(
                    (
                        art.TrajectoryGroup(rollout(openai_client, puzzle) for _ in range(12))
                        for puzzle in train_puzzles[batch_start_idx:batch_end_idx]
                    ),
                    pbar_desc=f"train (epoch {epoch+1}, batch {batch+1})",
                ),
            )
            await model.log(val_groups)
            await model.delete_checkpoints()
            await model.train(
                train_groups,
                config=art.TrainConfig(learning_rate=5e-5),
            )

if __name__ == "__main__":
    asyncio.run(main())