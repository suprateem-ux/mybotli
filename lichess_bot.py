import berserk
import chess
import chess.engine
import time
import logging
import threading
import random
import os
import chess.engine
import urllib.request
import sys

# Configuration
TOKEN = os.getenv("LICHESS_API_TOKEN")
print(TOKEN)

 
STOCKFISH_PATH = "./engines/stockfish-windows-x86-64-avx2.exe" # Adjust if needed
if not os.path.exists(STOCKFISH_PATH):
    print("Stockfish not found! Downloading Stockfish 17...")

    # Correct URL for Stockfish 17
    url = "https://github.com/official-stockfish/Stockfish/releases/download/sf_17/stockfish-windows-x86-64-avx2.exe"

    os.makedirs("engines", exist_ok=True)
    urllib.request.urlretrieve(url, STOCKFISH_PATH)
    print("Stockfish 17 downloaded!")



# Logging setup
logging.basicConfig(
    filename="lichess_bot.log", 
    level=logging.DEBUG, 
    format="%(asctime)s [%(levelname)s] - %(message)s"
)

# Lichess API
session = berserk.TokenSession(TOKEN)
client = berserk.Client(session)

# call bot
def get_active_bots():
    """Fetches a list of currently online Lichess bots."""
    bot_ids = ["raspfish", "endogenetic-bot", "Nikitosik-ai", "botyuliirma", "exogenetic-bot"]
    bot_list = []

    try:
        for bot in bot_ids:
            user = client.users.get_by_id(bot)  # Fetch each bot individually
            if user and user.get("title") == "BOT" and user.get("online", False):
                bot_list.append(user['id'])  # Add only if it's a bot and online

    except Exception as e:
        print(f"Error fetching bot list: {e}")
        return []  # Return empty list on error

    return bot_list  # Return the list of active bots

def challenge_random_bot():
    """Challenges a random online bot to a rated game."""
    bot_list = get_active_bots()

    if not bot_list:
        print("No bots found online. Retrying in 30 seconds...")
        time.sleep(30)
        return

    # Pick a random bot
    opponent_bot = random.choice(bot_list)

    try:
        client.challenges.create(
            opponent_bot,
            rated=True,  # Only rated games
            clock_limit=180,  # 3 minutes (adjust as needed)
            clock_increment=2,  # 2-second increment
            variant="standard",  # Standard chess
            color="random"  # Random color
        )
        print(f"Challenged bot {opponent_bot} to a rated 3+2 game!")
    
    except Exception as e:
        print(f"Failed to challenge bot {opponent_bot}: {e}")

# Stockfish engine
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
# Extreme speed settings for hyperbullet
HYPERBULLET_NODES = 190000 # Extremely low for fast move generation
HYPERBULLET_DEPTH = 5 # Minimal depth to save time
HYPERBULLET_MOVE_OVERHEAD = 50 # Lower overhead for near-instant moves

# Optimized settings for blitz
BLITZ_NODES = 555000 # More nodes for better strength
BLITZ_DEPTH = 18 # Good depth for accuracy
BLITZ_MOVE_OVERHEAD = 200 # Normal buffer for time safety

# Maximum strength settings for rapid and longer games
RAPID_NODES = 800000 # Deep calculation for strongest play
RAPID_DEPTH = 22 # Maximum depth for precision
RAPID_MOVE_OVERHEAD = 230 # Safe time buffer to avoid blunders

CLASSICAL_NODES = 1000000
CLASSICAL_DEPTH = 25
CLASSICAL_MOVE_OVERHEAD = 200

def configure_engine_for_time_control(time_control):
    """Dynamically configure Stockfish settings based on game time."""

    if time_control <= 30:  # Hyperbullet mode (extreme speed)
        engine.configure({
            "Nodes": HYPERBULLET_NODES,
            "Depth": HYPERBULLET_DEPTH,
            "Move Overhead": HYPERBULLET_MOVE_OVERHEAD,
            "Threads": 1,  # Max 2 threads for efficiency
            "Ponder": False,  # Disable pondering for speed
            "Use NNUE": False,  # Disable NNUE for ultra-fast evaluation
            "MultiPV": 1,
            "Hash": 32,
            "Book File": "C:/Users/Admin/Downloads/torom-boti/torom-boti/Perfect2023.bin",
            "Best Book move": True,
            "Book Depth": 6,
            "Book Variety": 25,
            "min_time": 0.01,
            "max_time": 0.04,
            "SyzygyPath": "https://tablebase.lichess.ovh",
            "SyzygyProbeDepth": 1,
            "SyzygyProbeLimit": 7,
            "Syzygy50MoveRule": True,
            "SyzygyRule50": True,
            "Lichess Opening Explorer": True,
            "Prioritize Book File": True
        })

    elif time_control <= 300:  # Blitz mode (balance between speed and strength)
        engine.configure({
            "Nodes": BLITZ_NODES,
            "Depth": BLITZ_DEPTH,
            "Move Overhead": BLITZ_MOVE_OVERHEAD,
            "Threads": 3,  # Use 4 threads for better move selection
            "Ponder": True,  # Enable pondering for stronger play
            "Use NNUE": True,  # Enable NNUE for better evaluation
            "MultiPV": 1,
            "Hash": 5120,
            "Book File": "C:/Users/Admin/Downloads/torom-boti/torom-boti/Perfect2023.bin",
            "Best Book move": True,
            "Book Depth": 10,
            "Book Variety": 40,
            "min_time": 0.05,
            "SyzygyPath": "https://tablebase.lichess.ovh",
            "SyzygyProbeDepth": 1,
            "SyzygyProbeLimit": 7,
            "Syzygy50MoveRule": True,
            "SyzygyRule50": True,
            "Lichess Opening Explorer": True,
            "Prioritize Book File": True
        })

    elif time_control <= 600:  # Short rapid mode (balance between speed and strength)
        engine.configure({
            "Nodes": RAPID_NODES,
            "Depth": RAPID_DEPTH,
            "Move Overhead": RAPID_MOVE_OVERHEAD,
            "Threads": 4,  # Use 4 threads for better move selection
            "Ponder": True,  # Enable pondering for stronger play
            "Use NNUE": True,  # Enable NNUE for better evaluation
            "MultiPV": 1,
            "Hash": 5192,
            "Book File": "C:/Users/Admin/Downloads/torom-boti/torom-boti/Perfect2023.bin",
            "Best Book move": True,
            "Book Depth": 11,
            "Book Variety": 40,
            "min_time": 0.3,
            "SyzygyPath": "https://tablebase.lichess.ovh",
            "SyzygyProbeDepth": 1,
            "SyzygyProbeLimit": 7,
            "Syzygy50MoveRule": True,
            "SyzygyRule50": True,
            "Lichess Opening Explorer": True,
            "Prioritize Book File": True
        })

    else:  # Rapid and longer games (maximum strength)
        engine.configure({
            "Nodes": CLASSICAL_NODES,
            "Depth": CLASSICAL_DEPTH,
            "Move Overhead": CLASSICAL_MOVE_OVERHEAD,
            "Threads": 6,  # Use more threads for deep calculations
            "Ponder": True,  # Enable pondering
            "Use NNUE": True,  # Strongest evaluation
            "MultiPV": 1,
            "Hash": 6144,
            "Book File": "C:/Users/Admin/Downloads/torom-boti/torom-boti/Perfect2023.bin",
            "Best Book move": True,
            "Book Depth": 20,
            "Book Variety": 45,
            "SyzygyPath": "https://tablebase.lichess.ovh",
            "SyzygyProbeDepth": 6,
            "SyzygyProbeLimit": 7,
            "Syzygy50MoveRule": True,
            "SyzygyRule50": True,
            "Lichess Opening Explorer": True,
            "Prioritize Book File": True
        })

    
      
# Infinite loop to keep challenging bots
while True:
    challenge_random_bot()
    time.sleep(10) # Wait 10 seconds before sending the next challenge



# Call this function before making a move
configure_engine_for_time_control(game["clock"])
# Time Management Settings
OVERHEAD_BUFFER = 0.15 # Extra time buffer to avoid losing on time
MAX_THINK_TIME = 5 # Never think more than this per move
BULLET_THINK = 0.005
BLITZ_THINK = 0.2
RAPID_THINK = 1.5

# Determine think time per move
def get_time_control(clock,is_losing):
    if not clock:
        return RAPID_THINK # Default if no time control
    initial = clock.get("initial",0) 
    increment = clock.get("increment",0)
  
    total_time = initial + 40 * increment # Estimate for 40 moves
    remaining_time = clock.get("remaining",total_time)/1000

    if total_time < 180: # Bullet
        base_think = max(0.01, remaining_time * 0.05) - OVERHEAD_BUFFER
    elif total_time < 600: # Blitz
        base_think = max(0.15, remaining_time * 0.05)
    else:# Rapid/Classical
         base_think = RAPID_THINK
     
    if is_losing:
        base_think = base_think *(0.3 if remaining_time < 10 else 0.5)
    safe_think_time = min(base_think,remaining_time * 0.2)

    return max(0.05, safe_think_time - OVERHEAD_BUFFER)


  
# Play a game
# Start the bot
# Function to handle playing a game
import logging
import chess
import chess.engine

# Function to play a game
def play_game(game_id):
    logging.info(f"🎯 Game started: {game_id}")
    game = client.games.export(game_id)
    board = chess.Board()

    # Messages for players & spectators
    client.bots.post_message(game_id, "HELLO! HI ! BEST OF LUCK🔥 NECROMINDX is here! Buckle up for an exciting game!🚀 Welcome to the battlefield where AI meets strategy, physics fuels precision, math calculates the odds, and the universe watches as we clash in a cosmic game of intellect! ♟️⚡🌌🤖📐
 🤖")
    client.bots.post_message(game_id, "🎭 Welcome, spectators! Watch NECROMINDX, built by @Suprateem11, in action!")

    move_time = get_time_control(game["clock"], False) - OVERHEAD_BUFFER

    try:
        while not board.is_game_over():
            try:
                result = engine.play(board, chess.engine.Limit(time=move_time))
                move = result.move.uci()
                client.bots.make_move(game_id, move)
                board.push(result.move)

                # ✅ Optimized logging
                logging.info(f"♟️ Move: {move} | ⏳ Time used: {move_time:.2f}s | FEN: {board.fen()}")

            except Exception as e:
                logging.error(f"🚨 Move Error: {e} | Board FEN: {board.fen()}")
                return  # Exit function instead of break to stop gracefully

    except Exception as e:
        logging.critical(f"🔥 Critical error in game loop: {e}")

    # Handle game result
    result = board.result()
    messages = {
        "1-0": "🏆 GG! You need some practice! I won this time! Thanks for playing! 😊",
        "0-1": "🤝 Well played! You got me this time. GG! 👍",
        "1/2-1/2": "⚖️ That was a solid game! A draw this time. 🤝"
    }

    client.bots.post_message(game_id, messages.get(result, "Game over!"))
    logging.info(f"📌 Game {game_id} finished with result: {result}")

# Function to handle Lichess events
def handle_events():
    """Listens for and handles incoming Lichess events."""
    try:
        for event in client.bots.stream_incoming_events():
            if event["type"] == "challenge":
                challenge = event["challenge"]
                if challenge["rated"]:
                    client.bots.accept_challenge(challenge["id"])
                    logging.info(f"✅ Accepted rated challenge from {challenge['challenger']['id']} ({challenge['timecontrol']['show']})")             
                else:
                    client.bots.decline_challenge(challenge["id"])
                    logging.info(f"❌ Declined unrated challenge from {challenge['challenger']['id']}")

            elif event.get("type") == "gameStart":
                try:
                    play_game(event["game"]["id"])  
                except Exception as e:
                    logging.error(f"🚨 Error in play_game: {e}")

    except Exception as e:
        logging.critical(f"🔥 Critical error in event loop: {e}")

if __name__ == "__main__":
    logging.info("Bot started...")
    stop_event = threading.Event()
    # Start event handling in a separate thread
    event_thread = threading.Thread(target=handle_events, daemon=True)
    event_thread.start()

    try:
        while not stop_event.is_set():
            time.sleep(1)  # Keep the main thread alive
    except KeyboardInterrupt:
        logging.info("Shutting down bot gracefully...")
        stop_event.set()  # Signal the event loop to stop
        event_thread.join()  # Wait for the thread to exit
        logging.info("Bot stopped cleanly.")
        sys.exit(0)