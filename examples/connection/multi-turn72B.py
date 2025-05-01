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
    name="003",
    project="multi-turn-connection",
    base_model="Qwen/Qwen2.5-72B-Instruct",
    _internal_config={"init_args": {"gpu_memory_utilization": 0.775}},
)

CONNECTION_SYSTEM_PROMPT = """
You are an expert Connections player. You will be presented with 16 words.
Your task is to find groups of four words that share a common connection.
There are exactly 4 groups to find. Each word belongs to only one group.

The game proceeds in turns. On each turn:
1. Analyze the remaining words.
2. Identify a group of four words you believe share a connection.

Provide your guess in JSON format within <output></output> tags as follows:
<output>
 {
    "category": "category name",
    "words": ["word1", "word2", "word3", "word4"]
  }
</output>

After each guess, you will receive feedback:
- If your guess is correct, you'll be told "Correct", and the words will be removed from the list.
- If your guess is incorrect, you'll be told "Incorrect", and the words will not be removed from the list.

You have a maximum of 4 mistakes allowed.
Continue guessing until you have found all 4 groups or run out of mistakes.
Only guess words currently available in the list of remaining words.
"""

def extract_turn_guess(content: str) -> Dict | None:
    """Extract and parse the model's guess for a single turn."""
    output_match = re.search(r'<output>(.*?)</output>', content, re.DOTALL)
    if not output_match:
        print("Warning: Cannot find <output> tags.")
        return None

    try:
        guess_data = json.loads(output_match.group(1).strip())
        # Basic validation
        if isinstance(guess_data, dict) and \
           "category" in guess_data and isinstance(guess_data["category"], str) and \
           "words" in guess_data and isinstance(guess_data["words"], list) and \
           len(guess_data["words"]) == 4 and all(isinstance(w, str) for w in guess_data["words"]) and \
           len(set(guess_data["words"])) == 4: # Ensure all 4 words are distinct
            return guess_data
        else:
            print(f"Warning: Invalid JSON structure or duplicate words in output: {guess_data}")
            return None
    except json.JSONDecodeError:
        print("Warning: Failed to parse JSON output.")
        return None


async def rollout(client: openai.AsyncOpenAI, puzzle: ConnectionPuzzle) -> art.Trajectory:
    """
    Run a model rollout on a Connection puzzle through multiple turns and evaluate the results.

    Args:
        client: OpenAI client for API calls
        puzzle: The Connection puzzle to solve

    Returns:
        Trajectory with evaluation metrics.
    """
    max_mistakes = 4
    mistakes = 0
    game_error = False # Flag for critical errors like invalid format or words

    remaining_words = {word.lower() for word in puzzle['words']}
    remaining_categories = [
        {"color": color.lower(), "name": name, "words": set(word.lower() for word in words)}
        for color, name, words in puzzle['solution']
    ]
    found_categories_details = [] # Store details of correctly guessed categories

    # Initialize trajectory without reward yet
    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": CONNECTION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Let's start! Here are the 16 words:\n{', '.join(list(remaining_words))}"
            }
        ],
        reward=0,
    )

    while len(found_categories_details) < 4 and mistakes < max_mistakes:
        # Get model response
        messages = get_trajectory_messages(trajectory)

        try:
            chat_completion = await client.chat.completions.create(
                messages=messages,
                model=model.name,
                max_tokens=2048
            )
            choice = chat_completion.choices[0]
            trajectory.messages_and_choices.append(choice)
            content = choice.message.content
            if not content: # Handle potential empty content
                 print("Warning: Model returned empty content.")
                 game_error = True
                 break

        except Exception as e:
            print(f"Error during API call: {e}")
            game_error = True
            break # Exit loop on API error

        # Validation
        guess = extract_turn_guess(content)
        if not guess:
            # Critical error: Invalid format
            game_error = True
            break

        guessed_words = set(word.lower() for word in guess['words'])
        # Ensure all guessed words exist in the remaining words
        if not guessed_words.issubset(remaining_words):
             # Critical error: Guessed invalid or already used words
             print(f"Game Error: Guessed extra/invalid words: {guessed_words - remaining_words}")
             game_error = True
             break

        correct_guess = False
        feedback = ""

        # Check if the guess matches any remaining category
        found_category_index = -1
        for i, category in enumerate(remaining_categories):
            if guessed_words == category["words"]:
                correct_guess = True
                found_category_index = i
                break # Exit the loop once a match is found

        if correct_guess:
            matched_category = remaining_categories.pop(found_category_index)
            found_categories_details.append(matched_category)
            remaining_words -= guessed_words
            feedback = f"Correct!"
        else:
            mistakes += 1
            feedback = f"Incorrect. Mistakes remaining: {max_mistakes - mistakes}"

        # Prepare next user message with feedback and remaining words, if the game is not over
        if len(found_categories_details) < 4 and mistakes < max_mistakes:
            feedback += f"\nRemaining words:\n{', '.join(list(remaining_words))}"
            trajectory.messages_and_choices.append(
                {"role": "user", "content": feedback}
            )
        # No need for an explicit game_over check here, the loop condition handles it

    if game_error:
        trajectory.reward = -5 # Penalize critical errors heavily
        trajectory.metrics = {"game_error": 1}
    else:
        # Calculate reward based on performance
        reward = len(found_categories_details) * 2.5 - mistakes * 0.5
        trajectory.reward = reward

        # Initialize base metrics
        metrics = {
            "num_correct": len(found_categories_details),
            "num_mistakes": mistakes,
            "game_error": 0,
        }

        # Initialize all color flags to 0
        color_metrics = {color: 0 for color in ["yellow", "green", "blue", "purple"]}

        # Update flags to 1 for found categories
        found_color_metrics = {cat['color'].lower(): 1 for cat in found_categories_details}
        color_metrics.update(found_color_metrics) # Overwrite 0s with 1s for found colors

        # Combine base metrics and color metrics
        metrics.update(color_metrics)
        trajectory.metrics = metrics

    if random.random() < 0.05:
        # print("Trajectory: ", trajectory) # Removed old trajectory print
        # print("Messages:", trajectory.messages()[1:]) # Removed old messages print

        # --- New Pretty Printing Logic ---
        print("\n--- Game Start ---")
        # Print initial words (from the first user message)
        initial_user_message = trajectory.messages_and_choices[1]
        if isinstance(initial_user_message, dict) and initial_user_message.get("role") == "user":
             print(f"Initial Words: {initial_user_message['content'].split(':')[-1].strip()}")
        else:
             # Fallback if structure is unexpected
             print("Initial Words:", list(remaining_words)) # Print initial state if message format changes

        turn_number = 1
        # Iterate through turns (assistant guess + user feedback)
        # Start from index 2 (first assistant message)
        for i in range(2, len(trajectory.messages_and_choices), 2):
            assistant_choice = trajectory.messages_and_choices[i]
            user_feedback_message = trajectory.messages_and_choices[i+1] if (i+1) < len(trajectory.messages_and_choices) else None

            print(f"\nTurn {turn_number}:")

            # Print Assistant Guess
            if hasattr(assistant_choice, 'message') and assistant_choice.message.content:
                content = assistant_choice.message.content
                guess = extract_turn_guess(content)
                if guess:
                    print(f"  Guess: Category='{guess['category']}', Words={guess['words']}")
                else:
                    # Handle cases where extraction fails or format is wrong mid-game
                    print(f"  Guess: (Invalid Format or Extraction Failed)")
                    print(f"    Raw Output: {content[:100]}...") # Print snippet of raw output
            else:
                print("  Guess: (No content found)")


            # Print User Feedback
            if isinstance(user_feedback_message, dict) and user_feedback_message.get("role") == "user":
                feedback_lines = user_feedback_message['content'].split('\n')
                print(f"  Feedback: {feedback_lines[0]}") # Print Correct/Incorrect line
                # Check if there are enough lines and the second line indicates remaining words
                if len(feedback_lines) > 2 and feedback_lines[1].strip() == "Remaining words:":
                     # Print the third line which contains the actual words
                     print(f"  Remaining words: {feedback_lines[2]}")
            elif user_feedback_message is None:
                 print("  Feedback: (Game ended after this guess)")
            else:
                 print("  Feedback: (Unexpected format)")


            turn_number += 1

        print("--- Game End ---")
        print(f"Final Result: Correct={len(found_categories_details)}, Mistakes={mistakes}, Game Error={game_error}")
        print(f"Reward: {trajectory.reward}")
        print(f"Metrics: {trajectory.metrics}\n")
        # --- End New Pretty Printing Logic ---


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
