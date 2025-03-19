import berserk
import chess
import logging
import threading
import random
import os
import chess.engine
import urllib.request
import sys
import psutil
import asyncio

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
    """🔥 The absolute peak of backoff brilliance 🔥"""
    max_retries = 7  
    base_delay = 5  
    backoff_factor = 2  
    max_wait_time = 300  

    retries = 0
    while retries < max_retries:
        bot_list = get_active_bots()

        if not bot_list:
            wait_time = min(base_delay * (backoff_factor ** retries), max_wait_time)
            jitter = random.uniform(-0.2 * wait_time, 0.2 * wait_time)
            final_wait_time = max(5, wait_time + jitter)

            logging.debug(f"⚠️ No bots found. Retrying in {final_wait_time:.1f}s (Attempt {retries + 1}/{max_retries})")
            time.sleep(final_wait_time)
            retries += 1
            continue

        retries = 0  # Reset retries since bots are available
        opponent_bot = random.choice(bot_list)

        try:
            client.challenges.create(
                opponent_bot,
                rated=True,
                clock_limit=180,
                clock_increment=2,
                variant="standard",
                color="random"
            )
            logging.debug(f"✅ Successfully challenged bot {opponent_bot} to a rated 3+2 game! 🚀")
            return  

        except Exception as e:
            logging.debug(f"❌ Challenge failed against {opponent_bot}: {e} (Retry {retries + 1}/{max_retries})")
            retries += 1
            time.sleep(10)  

    logging.debug("🚨 Max retries reached. No more challenges.")

# Stockfish engine
import chess.engine
import logging

import time

# Stockfish engine
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

# Dynamically determine system capabilities
TOTAL_RAM = psutil.virtual_memory().total // (1024 * 1024)  # Convert to MB
CPU_CORES = psutil.cpu_count(logical=False)

# Define optimized Stockfish settings
ENGINE_CONFIGS = {
    "hyperbullet": {
        "Nodes": 200000,
        "Depth": 5,
        "Move Overhead": 40,
        "Threads": max(1, CPU_CORES // 4),
        "Ponder": False,
        "Use NNUE": False,
        "MultiPV": 1,
        "Hash": min(64, TOTAL_RAM // 4),
        "Book File": "C:/Users/Admin/Downloads/torom-boti/torom-boti/Perfect2023.bin",
        "Best Book move": True,
        "Book Depth": 6,
        "Book Variety": 25,
        "SyzygyProbeDepth": min(1, TOTAL_RAM // 8192),
        "SyzygyProbeLimit": 7,
        "AutoLagCompensation": True,
        "BlunderDetection": True
    },
    "blitz": {
        "Nodes": 600000,
        "Depth": 20,
        "Move Overhead": 180,
        "Threads": max(2, CPU_CORES // 3),
        "Ponder": True,
        "Use NNUE": True,
        "MultiPV": 2,
        "Hash": min(512, TOTAL_RAM // 2),
        "Book File": "C:/Users/Admin/Downloads/torom-boti/torom-boti/Perfect2023.bin",
        "Best Book move": True,
        "Book Depth": 12,
        "Book Variety": 40,
        "SyzygyProbeDepth": min(2, TOTAL_RAM // 8192),
        "AutoLagCompensation": True
    },
    "rapid": {
        "Nodes": 900000,
        "Depth": 24,
        "Move Overhead": 250,
        "Threads": max(3, CPU_CORES // 2),
        "Ponder": True,
        "Use NNUE": True,
        "MultiPV": 3,
        "Hash": min(4096, TOTAL_RAM // 1.5),
        "Book File": "C:/Users/Admin/Downloads/torom-boti/torom-boti/Perfect2023.bin",
        "Best Book move": True,
        "Book Depth": 15,
        "Book Variety": 45,
        "SyzygyProbeDepth": min(4, TOTAL_RAM // 8192),
        "AutoLagCompensation": True
    },
    "classical": {
        "Nodes": 1200000,
        "Depth": 28,
        "Move Overhead": 300,
        "Threads": max(4, CPU_CORES),
        "Ponder": True,
        "Use NNUE": True,
        "MultiPV": 4,
        "Hash": min(8192, TOTAL_RAM),
        "Book File": "C:/Users/Admin/Downloads/torom-boti/torom-boti/Perfect2023.bin",
        "Best Book move": True,
        "Book Depth": 20,
        "Book Variety": 50,
        "SyzygyProbeDepth": min(6, TOTAL_RAM // 8192),
        "AutoLagCompensation": True
    }
}

def configure_engine_for_time_control(time_control):
    """Dynamically configure Stockfish settings based on game time."""
    if time_control <= 30:
        config = ENGINE_CONFIGS["hyperbullet"]
    elif time_control <= 300:
        config = ENGINE_CONFIGS["blitz"]
    elif time_control <= 600:
        config = ENGINE_CONFIGS["rapid"]
    else:
        config = ENGINE_CONFIGS["classical"]
    
    engine.configure(config)
    logging.debug(f"🔥 Stockfish configured for {time_control}s games: {config}")

    # Auto-Healing Mechanism
    try:
        engine.ping()
    except Exception as e:
        logging.error(f"⚠️ Engine crashed! Restarting... {e}")
        time.sleep(1)
        global engine
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        engine.configure(config)
        logging.info("✅ Stockfish restarted successfully!")

  
    
      
# Infinite loop to keep challenging bots
async def send_challenge():
    """Attempts to send a challenge and handles errors."""
    try:
        challenge_random_bot()  # Your function to send a challenge
        delay = random.uniform(8, 12)  # Randomized delay (anti-detection)
        logging.info(f"✅ Challenge sent! Next challenge in {delay:.2f} seconds...")
        return delay
    except Exception as e:
        logging.error(f"❌ Challenge failed: {e}")
        return 15  # Extra wait time after failure

async def challenge_loop():
    """Continuously sends challenges with adaptive delays."""
    failure_count = 0  # Track consecutive failures

    while True:
        delay = await send_challenge()
        
        # Exponential backoff on repeated failures
        if delay == 15:
            failure_count += 1
            backoff = min(60, 15 * (2 ** failure_count))  # Max wait 60 sec
            logging.warning(f"🔄 Retrying in {backoff} seconds due to failures...")
            await asyncio.sleep(backoff)
        else:
            failure_count = 0  # Reset failure count on success
            await asyncio.sleep(delay)

# Start the challenge loop asynchronously
asyncio.run(challenge_loop())


# Call this function before making a move
configure_engine_for_time_control(game["clock"])
# Time Management Settings
# ULTIMATE QUANTUM-AI TIME MANAGEMENT SYSTEM
OVERHEAD_BUFFER = 0.12  # Nano-optimized buffer to prevent flagging
MAX_THINK_TIME = 4.5  # Absolute maximum per move
PHASE_BOOST = 1.3  # Advanced adjustment for game phases
MOMENTUM_FACTOR = 1.4  # Extra time for critical moments
ANTI-TILT_FACTOR = 1.2  # Adjusts time to recover from bad positions

# Optimized base think time per game mode
THINK_TIME = {
    "bullet": 0.01,
    "blitz": 0.15,
    "rapid": 1.0,
    "classical": 3.2
}

def get_time_control(clock, is_losing, position_complexity, opponent_speed, game_phase):
    """Quantum-AI-driven time management for absolute dominance."""
    if not clock:
        return THINK_TIME["rapid"]  # Default to rapid if no time control
    
    initial = clock.get("initial", 0)
    increment = clock.get("increment", 0)
    total_time = initial + 40 * increment  # Estimate for 40 moves
    remaining_time = clock.get("remaining", total_time) / 1000  # Convert to seconds

    # Determine base think time based on total game length
    if total_time < 180:
        base_think = THINK_TIME["bullet"]  
    elif total_time < 600:
        base_think = THINK_TIME["blitz"]  
    elif total_time < 1800:
        base_think = THINK_TIME["rapid"]  
    else:
        base_think = THINK_TIME["classical"]  

    # Adjust for losing positions (defensive recalibration)
    if is_losing:
        base_think *= 0.3 if remaining_time < 10 else 0.55  

    # Boost think time in highly complex positions (ensures best moves)
    if position_complexity > 0.7:
        base_think *= PHASE_BOOST  

    # Game phase-based optimization (openings, middlegame, endgame)
    if game_phase == "opening":
        base_think *= 1.1  # More depth in theory-heavy positions
    elif game_phase == "endgame":
        base_think *= 1.5  # Endgames need precise calculations

    # Adjust for opponent speed (forces mistakes by altering tempo)
    if opponent_speed < 1.5:
        base_think *= 0.75  

    # Ensure we never exceed 20% of remaining time
    safe_think_time = min(base_think * MOMENTUM_FACTOR, remaining_time * 0.2)

    return max(0.05, safe_think_time - OVERHEAD_BUFFER)


  
# Play a game
# Start the bot
# Function to handle playing a game
# Function to play a game
def play_game(game_id):
    logging.info(f"🎯 Game started: {game_id}")
    
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
    logging.info("🚀 Bot initializing...")

    stop_event = threading.Event()

    def monitor_threads():
        """ Monitors if threads are alive and logs any issues. """
        while not stop_event.is_set():
            if not event_thread.is_alive():
                logging.error("🔥 Critical: Event thread stopped unexpectedly! Restarting...")
                restart_bot()  # Auto-restart if failure detected
            time.sleep(5)  # Check thread health every 5 seconds

    # Start event handling in a separate thread
    event_thread = threading.Thread(target=handle_events, daemon=True)
    event_thread.start()

    # Start thread monitoring
    monitor_thread = threading.Thread(target=monitor_threads, daemon=True)
    monitor_thread.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.1)  # Reduced latency for better real-time performance
    except KeyboardInterrupt:
        logging.info("🛑 Graceful shutdown initiated...")
        stop_event.set()  # Signal threads to stop
        event_thread.join(timeout=3)  # Ensure clean exit
        monitor_thread.join(timeout=3)
        logging.info("✅ Bot stopped cleanly. All systems shut down.")
        sys.exit(0)
