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
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
import lichess
import time

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
    bot_ids = ["raspfish", "endogenetic-bot", "Nikitosik-ai", "botyuliirma", "exogenetic-bot","EnergyOfBeingBot"]
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
    """Attempts to send a challenge while avoiding detection."""
    try:
        challenge_random_bot()  # Function to send a challenge
        delay = random.uniform(8, 12) + random.uniform(-2, 2)  # Natural variation
        logging.info(f"✅ Challenge sent! Next challenge in {delay:.2f} seconds...")
        return delay
    except Exception as e:
        logging.error(f"❌ Challenge failed: {e}")
        return 15  # Extra wait time after failure

async def challenge_loop():
    """Continuously sends challenges while adapting to failures."""
    failure_count = 0
    total_failures = 0

    while True:
        delay = await send_challenge()

        if delay == 15:  # Challenge failed
            failure_count += 1
            total_failures += 1

            # **Smart exponential backoff** (max 90 sec wait)
            backoff = min(90, 15 * (2 ** failure_count))
            logging.warning(f"🔄 Retrying in {backoff} seconds due to failures...")
            await asyncio.sleep(backoff)

            # **Stealth Cloaking Mode** - If too many failures, bot **vanishes temporarily**
            if failure_count >= 3:
                stealth_cooldown = random.randint(300, 900)  # 5-15 minutes
                logging.error(f"🕵️ Cloaking Mode ON: Cooling down for {stealth_cooldown} seconds...")
                await asyncio.sleep(stealth_cooldown)
                failure_count = 0  # Reset failure count
            
            # **Emergency Anti-Ban Mode** - Long cool-down to avoid Lichess bans
            if total_failures >= 10:
                ultra_cooldown = random.randint(1800, 3600)  # 30-60 min cooldown
                logging.critical(f"🚨 Lichess Anti-Ban Mode ACTIVATED. Cooling down for {ultra_cooldown} seconds...")
                await asyncio.sleep(ultra_cooldown)
                total_failures = 0  # Reset total failures
        else:
            failure_count = 0  # Reset failure streak on success
            jitter = random.uniform(-3, 3)  # Makes behavior unpredictable
            await asyncio.sleep(delay + jitter)

# Start the challenge loop asynchronously
asyncio.run(challenge_loop())
# Call this function before making a move
if "clock" in game:
    configure_engine_for_time_control(game["clock"])

# TIME MANAGEMENT SYSTEM 🚀♟️
# The most insane Quantum-AI-driven time control system ever. 

# Hyper-optimized settings for ultimate performance
OVERHEAD_BUFFER = 0.07  # Ultra-precise buffer to avoid flagging
MAX_THINK_TIME = 5.5  # Absolute maximum time per move
PHASE_BOOST = 1.45  # Extra calculation for complex positions
MOMENTUM_FACTOR = 1.6  # Boosts time when attacking
ANTI_TILT_FACTOR = 1.35  # Prevents tilt by adjusting timing dynamically
ENDGAME_BOOST = 2.0  # Maximum precision in critical endgames
SPEED_ADJUSTMENT = 0.6  # Adapts based on opponent's move speed
AGGRESSIVE_MODE = 1.3  # Expands time when in winning positions
DEFENSE_MODE = 0.5  # Conserves time when in losing positions
TEMPO_PRESSURE = 0.8  # Forces mistakes by playing faster at key moments

# Optimized base think time for each time control format
THINK_TIME = {
    "bullet": 0.007,  # Minimal time per move in bullet
    "blitz": 0.1,  # Slightly increased for blitz
    "rapid": 0.85,  # Deeper calculations in rapid
    "classical": 3.8  # Maximum depth in classical
}

def get_time_control(clock, is_losing, position_complexity, opponent_speed, game_phase):
    """INSANE Quantum-AI-driven time management for absolute domination."""
    if not clock:
        return THINK_TIME["rapid"]  # Default to rapid if no time control provided
    
    initial = clock.get("initial", 0)
    increment = clock.get("increment", 0)
    total_time = initial + 40 * increment  # Estimated total time for 40 moves
    remaining_time = clock.get("remaining", total_time) / 1000  # Convert ms to seconds

    # Base think time based on game duration
    if total_time < 180:
        base_think = THINK_TIME["bullet"]  
    elif total_time < 600:
        base_think = THINK_TIME["blitz"]  
    elif total_time < 1800:
        base_think = THINK_TIME["rapid"]  
    else:
        base_think = THINK_TIME["classical"]  

    # Defensive mode: Adjust time if losing
    if is_losing:
        base_think *= DEFENSE_MODE if remaining_time < 10 else 0.55  

    # Complexity scaling: Allocate more time in sharp positions
    if position_complexity > 0.75:
        base_think *= PHASE_BOOST  

    # Game phase optimizations
    if game_phase == "opening":
        base_think *= 1.2  # Extra depth in opening prep
    elif game_phase == "middlegame":
        base_think *= MOMENTUM_FACTOR  # Prioritize deep calculations in the fight
    elif game_phase == "endgame":
        base_think *= ENDGAME_BOOST  # Maximum precision in winning positions

    # Opponent speed adjustments: Adapt to their tempo
    if opponent_speed < 1.0:  # Slow opponent, use time wisely
        base_think *= 1.25  
    elif opponent_speed > 2.0:  # Speedster detected, play fast to match tempo
        base_think *= SPEED_ADJUSTMENT  

    # Aggressive mode: More time when in a clearly winning position
    if remaining_time > total_time * 0.45:
        base_think *= AGGRESSIVE_MODE  

    # Tempo pressure: Force mistakes by adjusting move speed dynamically
    if remaining_time < total_time * 0.25:  # When time is low, play faster
        base_think *= TEMPO_PRESSURE  

    # Ensure bot never exceeds 20% of remaining time
    safe_think_time = min(base_think * MOMENTUM_FACTOR, remaining_time * 0.2)

    return max(0.05, safe_think_time - OVERHEAD_BUFFER)


  
# Play a game
# Start the bot
# Function to handle playing a game
# Function to play a game
logger.add("lichess_bot.log", rotation="10 MB", retention="1 month", level="DEBUG")

# 🚀 Global Async Event Loop & Threading
stop_event = threading.Event()
executor = ThreadPoolExecutor(max_workers=10)  # Smart Auto-Scaling Threading

async def play_game(game_id):
    """ AI-Optimized Game Play with Parallel Move Calculation & Intelligent Messaging """
    logger.info(f"🎯 Game started: {game_id}")
    
    client.bots.post_message(game_id, "HELLO! HI! BEST OF LUCK🔥 NECROMINDX is here! Buckle up for an exciting game!🚀Welcome to the battlefield where AI meets strategy, quantum physics fuels precision, math calculates the odds, and the universe watches as we clash in a cosmic game of intellect! ♟️⚡🌌🤖📐
 🤖")
    client.bots.post_message(game_id, "🎭 Welcome, spectators! Watch NECROMINDX, built by @Suprateem11, in ultimate quantum action!")

    board = chess.Board()
    move_time = get_time_control(game["clock"], False) - OVERHEAD_BUFFER

    try:
        while not board.is_game_over():
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    executor, lambda: engine.play(board, chess.engine.Limit(time=move_time))
                )
                move = result.move.uci()
                client.bots.make_move(game_id, move)
                board.push(result.move)

                # ✅ Optimized logging
                logger.info(f"♟️ Move: {move} | ⏳ Time used: {move_time:.2f}s | FEN: {board.fen()}")

            except Exception as e:
                logger.error(f"🚨 Move Error: {e} | Board FEN: {board.fen()}")
                return  # Exit function instead of break to stop gracefully

    except Exception as e:
        logger.critical(f"🔥 Critical error in game loop: {e}")

    # Handle game result
    result = board.result()
    messages = {
        "1-0": "🏆 GG! You need some practice! I won this time! Thanks for playing! 😊",
        "0-1": "🤝 Well played! You got me this time. GG! 👍",
        "1/2-1/2": "⚖️ That was a solid game! A draw this time. 🤝"
    }

    client.bots.post_message(game_id, messages.get(result, "Game over!"))
    logger.info(f"📌 Game {game_id} finished with result: {result}")

async def handle_events():
    """ Fully Asynchronous, AI-Powered Lichess Event Handling """
    try:
        async for event in client.bots.stream_incoming_events():
            if event["type"] == "challenge":
                challenge = event["challenge"]
                if challenge["rated"]:
                    await client.bots.accept_challenge(challenge["id"])
                    logger.info(f"✅ Accepted rated challenge from {challenge['challenger']['id']}")
                else:
                    await client.bots.decline_challenge(challenge["id"])
                    logger.info(f"❌ Declined unrated challenge")

            elif event.get("type") == "gameStart":
                asyncio.create_task(play_game(event["game"]["id"]))

    except Exception as e:
        logger.critical(f"🔥 Critical error in event loop: {e}")

def monitor_threads():
    """ 🚨 AI-Powered Thread Monitoring & Auto-Healing """
    while not stop_event.is_set():
        if not event_thread.is_alive():
            logger.error("🔥 CRITICAL: Event thread stopped! Restarting...")
            restart_bot()
        time.sleep(0.5)  # Ultra-Fast Health Checks

if __name__ == "__main__":
    logger.info("🚀 Lichess Bot Starting... AI Mode Activated")

    # Start event handling in a separate thread
    event_thread = threading.Thread(target=lambda: asyncio.run(handle_events()), daemon=True)
    event_thread.start()

    # Start AI-powered thread monitoring
    monitor_thread = threading.Thread(target=monitor_threads, daemon=True)
    monitor_thread.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.05)  # Lightning-Fast Event Loop
    except KeyboardInterrupt:
        logger.info("🛑 Graceful Shutdown Initiated...")
        stop_event.set()
        event_thread.join(timeout=3)
        monitor_thread.join(timeout=3)
        logger.info("✅ Bot Stopped Cleanly.")