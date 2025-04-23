import art
import asyncio
from dotenv import load_dotenv
import json
import openai
import random
import re
from typing import TypedDict, List, Tuple, Literal, Dict, Set
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


# Define the model
model = art.TrainableModel(
    name="010",
    project="connection",
    base_model="Qwen/Qwen2.5-14B-Instruct",
    _internal_config={"init_args": {"gpu_memory_utilization": 0.775}},
)


# System prompt for the Connection puzzle game
CONNECTION_SYSTEM_PROMPT = """
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

First, generate 3 possible combinations with reasoning within <exploration></exploration> tags:

<exploration>
Possibility 1:
[Describe your first possible grouping of the words into 4 categories]

Possibility 2:
[Describe your second possible grouping of the words into 4 categories]

Possibility 3:
[Describe your third possible grouping of the words into 4 categories]
</exploration>

Then, think through these 3 possibilities and provide your final reasoning within <reasoning></reasoning> tags. You may use one of your existing solutions or create a new one based on your analysis.

<reasoning>
Explain your thought process for identifying each category and why the words belong together in maximum of 800 words.
</reasoning>

Finally, provide your solution in JSON format within <output></output> tags as follows:

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
"""


def extract_model_solution(content: str) -> List[Dict] | None:
    """Extract and parse the model's solution from the response content."""
    output_match = re.search(r'<output>(.*?)</output>', content, re.DOTALL)
    if not output_match:
        print("Cannot parse output")
        return None
    
    try:
        return json.loads(output_match.group(1).strip())
    except json.JSONDecodeError:
        print("Failed to parse JSON output")
        return None


def validate_model_solution(model_solution: List[Dict], puzzle_words: List[str]) -> bool:
    """Validate that the model's solution contains the correct words."""
    if not model_solution:
        return False
        
    # Extract words from the model's solution
    all_model_words = []
    for category in model_solution:
        if "words" in category and isinstance(category["words"], list):
            if len(category["words"]) != 4:
                print(f"Warning: Invalid number of words in category: {category}")
                return False
            all_model_words.append(category["words"])

    # Flatten the model's solution words
    flat_model_words = [word for category in all_model_words for word in category]
    
    # Check if the sets of words match
    if set(flat_model_words) != set(puzzle_words):
        print("Warning: Model solution contains different words than the puzzle")
        missing_from_model = set(puzzle_words) - set(flat_model_words)
        extra_in_model = set(flat_model_words) - set(puzzle_words)
        if missing_from_model:
            print(f"Missing from model: {missing_from_model}")
        if extra_in_model:
            print(f"Extra in model: {extra_in_model}")
        return False
    
    return True


def prepare_categories_for_matching(categories: List[Dict]) -> List[Dict]:
    """Convert categories to a format suitable for matching."""
    prepared_categories = []
    for category in categories:
        prepared_categories.append({
            "color": category["color"],
            "category": category["category"],
            "words": set(word.lower() for word in category["words"])
        })
    return prepared_categories


def prepare_puzzle_categories(solution: List[Tuple]) -> List[Dict]:
    """Convert puzzle solution to a format suitable for matching."""
    puzzle_categories = []
    for color, category_name, words in solution:
        puzzle_categories.append({
            "color": color,
            "category": category_name,
            "words": set(word.lower() for word in words)
        })
    return puzzle_categories


def perform_greedy_matching(puzzle_categories: List[Dict], model_categories: List[Dict]) -> List[Dict]:
    """Match puzzle categories to model categories using a greedy approach."""
    matched_model_indices = set()
    matches = []
    
    for puzzle_cat in puzzle_categories:
        best_match_score = -1
        best_match_idx = -1
        
        # Find the best matching model category that hasn't been matched yet
        for i, model_cat in enumerate(model_categories):
            if i in matched_model_indices:
                continue
            
            # Calculate intersection size (number of matching words)
            intersection = len(puzzle_cat["words"].intersection(model_cat["words"]))
            
            if intersection > best_match_score:
                best_match_score = intersection
                best_match_idx = i
        
        if best_match_idx != -1:
            matched_model_indices.add(best_match_idx)
            matches.append({
                "puzzle_category": puzzle_cat,
                "model_category": model_categories[best_match_idx],
                "match_score": best_match_score
            })
    
    return matches


def calculate_metrics(matches: List[Dict]) -> Tuple[float, float, Dict[str, float]]:
    """Calculate reward and accuracy metrics based on the matches."""
    weighted_correct = 0
    color_weights = {"yellow": 0.5, "green": 0.5, "blue": 0.5, "purple": 0.5}
    color_accuracies = {}
    reward = 0
    
    for match in matches:
        puzzle_cat = match["puzzle_category"]
        model_cat = match["model_category"]
        color = puzzle_cat["color"]
        
        # Calculate word accuracy
        correct_words = len(puzzle_cat["words"].intersection(model_cat["words"]))
        word_accuracy = correct_words / 4
        weighted_word_accuracy = color_weights[color] * word_accuracy
        
        if word_accuracy == 1:
            # All 4 words are correct
            reward += 1
        
        weighted_correct += weighted_word_accuracy
        color_accuracies[color] = word_accuracy
    
    # Calculate metrics
    max_weighted_score = sum(color_weights.values())
    weighted_accuracy = weighted_correct / max_weighted_score
    
    # Double reward if perfect accuracy
    if reward == 4:
        # All 4 categories are correct
        reward += 4
    
    reward += weighted_correct
    
    return reward, weighted_accuracy, color_accuracies


def log_results(content: str, reward: float, weighted_accuracy: float, 
                color_accuracies: Dict[str, float], matches: List[Dict] = None):
    """Log the results of the evaluation."""
    print(content)
    print(f"Reward: {reward:.2f}")
    print(f"Weighted accuracy: {weighted_accuracy:.2f}")
    print(f"Per-color accuracy: Yellow: {color_accuracies.get('yellow', 0):.2f}, "
          f"Green: {color_accuracies.get('green', 0):.2f}, "
          f"Blue: {color_accuracies.get('blue', 0):.2f}, "
          f"Purple: {color_accuracies.get('purple', 0):.2f}")
    
    # Optionally print matching details
    # if matches:
    #     print("Greedy matching results:")
    #     for match in matches:
    #         puzzle_cat = match["puzzle_category"]
    #         model_cat = match["model_category"]
    #         print(f"Puzzle {puzzle_cat['color']} ({puzzle_cat['category']}) matched with model {model_cat['color']} ({model_cat['category']})")
    #         print(f"  Matching words: {puzzle_cat['words'].intersection(model_cat['words'])}")
    #         print(f"  Score: {match['match_score']}/4")


async def rollout(client: openai.AsyncOpenAI, puzzle: ConnectionPuzzle) -> art.Trajectory:
    """
    Run a model rollout on a Connection puzzle and evaluate the results.
    
    Args:
        client: OpenAI client for API calls
        puzzle: The Connection puzzle to solve
        
    Returns:
        Trajectory with evaluation metrics
    """
    # Initialize trajectory
    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": CONNECTION_SYSTEM_PROMPT,
            }
        ],
        reward=0,
        metrics={"acc": 0}
    )
    
    # Add user message with puzzle words
    trajectory.messages_and_choices.append({
        "role": "user",
        "content": " ".join(puzzle["words"])
    })

    # Get model response
    messages = get_trajectory_messages(trajectory)
    chat_completion = await client.chat.completions.create(
        messages=messages,
        model=model.name,
        max_tokens=2048
    )
    choice = chat_completion.choices[0]
    trajectory.messages_and_choices.append(choice)

    content = choice.message.content
    
    # Extract and validate model solution
    model_solution = extract_model_solution(content)
    if not model_solution or not validate_model_solution(model_solution, puzzle["words"]):
        return trajectory
    
    # Prepare categories for matching
    model_categories = prepare_categories_for_matching(model_solution)
    puzzle_categories = prepare_puzzle_categories(puzzle["solution"])
    
    # Perform greedy matching
    matches = perform_greedy_matching(puzzle_categories, model_categories)
    
    # Calculate metrics
    reward, weighted_accuracy, color_accuracies = calculate_metrics(matches)
    
    # Update trajectory with metrics
    trajectory.reward = reward
    trajectory.metrics["acc"] = weighted_accuracy
    
    # Add per-color metrics
    for color, accuracy in color_accuracies.items():
        trajectory.metrics[f"acc_{color}"] = accuracy
    
    # Log results occasionally
    if random.random() < 0.05:
        log_results(content, reward, weighted_accuracy, color_accuracies)
    
    return trajectory
    
    
async def main():
    connection_puzzles: list[ConnectionPuzzle] = load_puzzles("examples/connection/puzzle.jsonl")
    print(connection_puzzles[0])

    # Shuffle all puzzles before splitting into train/val/test
    random.seed(42)
    random.shuffle(connection_puzzles)

    # Now split into train/val/test sets
    val_size = 50
    test_size = 50

    val_puzzles = connection_puzzles[:val_size]
    test_puzzles = connection_puzzles[val_size:val_size+test_size]
    train_puzzles = connection_puzzles[val_size+test_size:]

    # No need to shuffle train_puzzles again as all puzzles were already shuffled

    await model.register(art.LocalAPI())

    batch_size = 4
    num_epochs = 3
    openai_client = model.openai_client()
    
    start_step = await model.get_step()
    max_steps = 1000
    current_step = start_step
    
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}")
        # Shuffle training data at the beginning of each epoch
        random.shuffle(train_puzzles)
        
        # Calculate how many batches we can process in this epoch
        num_batches = len(train_puzzles) // batch_size
        
        for batch in range(num_batches):
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
                        art.TrajectoryGroup(rollout(openai_client, puzzle) for _ in range(24))
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
            
            current_step += 1
            if current_step >= max_steps:
                print(f"Reached max steps ({max_steps}). Stopping training.")
                break
        
        if current_step >= max_steps:
            break


if __name__ == "__main__":
    asyncio.run(main())
    
    
        
        
