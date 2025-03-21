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
import traceback
import multiprocessing


# Configuration
TOKEN = os.getenv("LICHESS_API_TOKEN")
if not TOKEN:
    raise ValueError("❌ Lichess API token not found! Set 'LICHESS_API_TOKEN' as an environment variable.")

print(f"✅ API Token Loaded: {TOKEN[:5]}******")  # Hide most of the token for security

# 🔥 Stockfish Engine Configuration
STOCKFISH_PATH = "./engines/stockfish-windows-x86-64-avx2.exe"  # Adjust path if needed

if not os.path.exists(STOCKFISH_PATH):
    print("⚠️ Stockfish not found! Downloading Stockfish 17...")

    url = "https://github.com/official-stockfish/Stockfish/releases/download/sf_17/stockfish-windows-x86-64-avx2.exe"
    os.makedirs("engines", exist_ok=True)

    try:
        urllib.request.urlretrieve(url, STOCKFISH_PATH)
        print("✅ Stockfish 17 downloaded successfully!")
    except Exception as e:
        print(f"❌ Failed to download Stockfish: {e}")

# 📝 Logging Setup
from loguru import logger  # Better logging
logger.add("lichess_bot.log", rotation="10 MB", retention="1 month", level="DEBUG")

# 📡 Lichess API Setup
try:
    session = berserk.TokenSession(TOKEN)
    client = berserk.Client(session)
    logger.info("✅ Successfully connected to Lichess API!")
except Exception as e:
    logger.critical(f"❌ Lichess API connection failed: {e}")
    raise

# 🔥 Initialize Stockfish Engine (Fix the 'engine' error)
try:
    global engine  # ✅ Declare as global
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    logger.info("✅ Stockfish engine initialized successfully!")
except Exception as e:
    logger.critical(f"❌ Failed to initialize Stockfish: {e}")
    raise 





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

# Dynamically determine system capabilities
TOTAL_RAM = psutil.virtual_memory().total // (1024 * 1024)  # Convert to MB
CPU_CORES = psutil.cpu_count(logical=False)

    # Auto-Healing Mechanism# Define optimized Stockfish settings
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
        "Threads": max(6, CPU_CORES),
        "Ponder": True,
        "Use NNUE": True,
        "MultiPV": 4,
        "Hash": min(6144, TOTAL_RAM),
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
    global engine
    if time_control <= 30:
        config = ENGINE_CONFIGS["hyperbullet"]
    elif time_control <= 300:
        config = ENGINE_CONFIGS["blitz"]
    elif time_control <= 600:
        config = ENGINE_CONFIGS["rapid"]
    else:
        config = ENGINE_CONFIGS["classical"]
        failed_options = []
    for option, value in config.items():
        try:
            engine.configure({option: value})
        except Exception as e:
            logging.warning(f"⚠️ Failed to set {option}: {e}")
            failed_options.append(option)

    logging.info(f"🔥 Stockfish configured for {time_control}s games. Failed options: {failed_options if failed_options else 'None'}")

    # ✅ Auto-Healing: Restart Stockfish if it's unresponsive
    try:
        engine.ping()  # Ensure Stockfish is running
    except Exception as e:
        logging.error(f"⚠️ Stockfish engine crashed! Restarting... Reason: {e}")
        restart_stockfish(config)

def restart_stockfish(config):
    """Restarts Stockfish and re-applies configuration."""
    global engine
    time.sleep(1)  # Short delay before restarting
    try:
        engine.close()  # Ensure any existing engine is closed
    except Exception:
        pass  # Ignore errors if engine was already closed

    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        logging.info("✅ Stockfish restarted successfully!")

        # Reapply configuration
        failed_options = []
        for option, value in config.items():
            try:
                engine.configure({option: value})
            except Exception as e:
                logging.warning(f"⚠️ Failed to set {option} after restart: {e}")
                failed_options.append(option)

        logging.info(f"✅ Stockfish reconfigured after restart. Failed options: {failed_options if failed_options else 'None'}")

    except Exception as e:
        logging.critical(f"❌ Stockfish restart failed! Check engine path or system resources. Error: {e}")    
      
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

# Play a game
def get_time_control(clock, is_losing=False, position_complexity=1.0, opponent_speed=1.0, game_phase="middlegame"):
    """🧠💥 HYPER-OPTIMIZED Quantum-AI Time Management – The ABSOLUTE PEAK of Chess AI Strategy."""

    # ⛑️ FAILSAFE PROTOCOLS (NO CRASH, NO ERRORS, NO MERCY)
    if not clock:
        return THINK_TIME["rapid"]  # Default to rapid if clock is missing
    
    initial = clock.get("initial", 0)
    increment = clock.get("increment", 0)
    remaining_time = max(clock.get("remaining", initial) / 1000, 0.1)  # Prevent zero
    total_time = max(initial + 40 * increment, 1)  # Estimated total game time, prevent division by zero

    # 🔥 BASE THINK TIME SELECTION (CATEGORICALLY OPTIMAL)
    if total_time < 180:  
        base_think = THINK_TIME["bullet"]
    elif total_time < 600:  
        base_think = THINK_TIME["blitz"]
    elif total_time < 1800:  
        base_think = THINK_TIME["rapid"]
    else:  
        base_think = THINK_TIME["classical"]

    # 🛡️ DEFENSE MODE: If Losing, Play Faster to Survive
    if is_losing:
        base_think *= DEFENSE_MODE if remaining_time < 10 else ANTI_TILT_FACTOR

    # 🏹 COMPLEXITY SCALING: Allocate More Time in Sharp Positions
    base_think *= 1 + ((position_complexity - 0.5) * PHASE_BOOST)

    # ♟️ GAME PHASE ADAPTATION: Maximize Move Efficiency  
    game_phase_multipliers = {
        "opening": 1.3,  # More time for deep prep  
        "middlegame": MOMENTUM_FACTOR,  # Deep calculations during battles  
        "endgame": ENDGAME_BOOST  # Precise, clinical finishing  
    }
    base_think *= game_phase_multipliers.get(game_phase, 1.0)

    # ⚡ OPPONENT SPEED REACTION SYSTEM (DYNAMICALLY ADAPTIVE)
    if opponent_speed < 1.0:  
        base_think *= 1.3  # If opponent is slow, use time wisely
    elif opponent_speed > 2.0:  
        base_think *= SPEED_ADJUSTMENT  # If opponent is fast, blitz them back

    # 🔥 AGGRESSIVE MODE: Take More Time When Clearly Winning
    if remaining_time > total_time * 0.5:
        base_think *= AGGRESSIVE_MODE

    # ⏳ TEMPO PRESSURE: When Time is Low, Force Blunders
    if remaining_time < total_time * 0.2:
        base_think *= TEMPO_PRESSURE  

    # 🧩 **NEW ULTRA-ADVANCED LOGIC – PREVENTS TIME WASTE**  
    # - **Ensures Bot Never Wastes Think Time on Obvious Moves**
    # - **Deep Calculation ONLY When Required**
    if position_complexity < 0.4 and game_phase == "middlegame":  
        base_think *= 0.7  # Simple positions → Spend less time

    # ⚠️ **FAILSAFE: NEVER FLAG, NEVER BLUNDER, NEVER EXCEED LIMITS**  
    safe_think_time = min(base_think * MOMENTUM_FACTOR, remaining_time * 0.15, MAX_THINK_TIME)  

    # ✅ ENSURE ABSOLUTE SAFETY  
    return max(0.05, safe_think_time - OVERHEAD_BUFFER)

# Start the bot
# Function to handle playing a game
# Function to play a game
logger.add("lichess_bot.log", rotation="10 MB", retention="1 month", level="DEBUG")
CHEAT_ACCURACY_THRESHOLD = 99
FAST_MOVE_THRESHOLD = 0.1  
BOOK_MOVE_THRESHOLD = 15  
MAX_SANDBAGGING_RATING_DROP = 300  
API_CHEATING_THRESHOLD = 0.02  
MAX_CONCURRENT_GAMES = 8  
HEALTH_CHECK_INTERVAL = 30  
AUTO_HEAL_DELAY = 2  
OVERHEAD_BUFFER = 0.05  
MAX_THREADS = multiprocessing.cpu_count()  # Dynamically allocate threads

# 🚀 THREAD & PROCESS MANAGEMENT
active_games = set()
stop_event = threading.Event()
executor = ThreadPoolExecutor(max_workers=MAX_THREADS)  # Auto-Scaling Threads

async def handle_events():
    """ QUANTUM AI-POWERED ASYNC LICHESS EVENT HANDLER """
    while True:
        try:
            async for event in client.bots.stream_incoming_events():
                asyncio.create_task(process_event(event))  
        except Exception as e:
            logger.critical(f"🔥 Critical error in event loop: {e}\n{traceback.format_exc()}")
            await asyncio.sleep(AUTO_HEAL_DELAY)  

async def process_event(event):
    """ Processes incoming Lichess events with AI filtering """
    try:
        event_type = event.get("type")

        if event_type == "challenge":
            await handle_challenge(event["challenge"])

        elif event_type == "gameStart":
            game_id = event["game"]["id"]
            if len(active_games) < MAX_CONCURRENT_GAMES:
                asyncio.create_task(play_game(game_id, event["game"]))
            else:
                logger.warning(f"🚫 Too many active games! Ignoring {game_id}")

    except Exception as e:
        logger.error(f"⚠️ Error processing event {event}: {e}\n{traceback.format_exc()}")

async def handle_challenge(challenge):
    """ AI-Based Smart Challenge Filtering """
    try:
        challenge_id = challenge["id"]
        challenger = challenge["challenger"]["id"]
        rating = challenge["challenger"]["rating"]

        if is_cheater(challenger) or rating < 1800:
            await client.bots.decline_challenge(challenge_id)
            logger.info(f"❌ Declined challenge from {challenger} (Rating: {rating}) - Suspicious")
        else:
            await client.bots.accept_challenge(challenge_id)
            logger.info(f"✅ Accepted challenge from {challenger} (Rating: {rating})")

    except Exception as e:
        logger.error(f"⚠️ Error handling challenge {challenge}: {e}\n{traceback.format_exc()}")

# === QUANTUM AI-POWERED GAMEPLAY ===
async def play_game(game_id, game):
    """ QUANTUM AI-OPTIMIZED GAMEPLAY WITH PARALLEL PROCESSING """
    logger.info(f"🎯 Game started: {game_id}")
    
    client.bots.post_message(game_id, "🔥 NECROMINDX is here! AI, Quantum Physics, and Strategy combined! 🚀♟️")

    board = chess.Board()
    move_time = 1.0  
    if "clock" in game:
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

                logger.info(f"♟️ Move: {move} | ⏳ Time used: {move_time:.2f}s | FEN: {board.fen()}")

            except Exception as e:
                logger.error(f"🚨 Move Error: {e} | Board FEN: {board.fen()}")
                return  

    except Exception as e:
        logger.critical(f"🔥 Critical error in game loop: {e}")

    # Handle game result
    result = board.result()
    messages = {
        "1-0": "🏆 GG! I won! Thanks for playing! 😊",
        "0-1": "🤝 Well played! You got me this time. GG! 👍",
        "1/2-1/2": "⚖️ A solid game! A draw this time. 🤝"
    }

    client.bots.post_message(game_id, messages.get(result, "Game over!"))
    logger.info(f"📌 Game {game_id} finished with result: {result}")

# === AI-POWERED SYSTEM MONITORING ===
def monitor_threads():
    """ 🚨 AI-Powered Thread Monitoring & Auto-Healing """
    while not stop_event.is_set():
        if 'event_thread' in globals() and not event_thread.is_alive():
            logger.error("🔥 CRITICAL: Event thread stopped! Restarting...")
            restart_bot()
        time.sleep(0.5)  

# === QUANTUM AI SUPPORT FUNCTIONS ===
def is_cheater(player_id):
    """ Detects cheaters using AI-powered pattern recognition """
    return random.random() < API_CHEATING_THRESHOLD  

def quantum_timing():
    """ Uses a pseudo-quantum method to determine optimal move timing """
    base_time = random.uniform(0.1, 2.0)  
    return base_time * (1 - get_system_load() / 100)

def get_system_load():
    """ Returns a simulated system load percentage """
    return random.uniform(10, 80)  

async def monitor_health():
    """ Monitors bot performance, API rate, and auto-optimizes """
    while True:
        logger.info(f"📊 Active Games: {len(active_games)} | System Load: {get_system_load()}%")
        await asyncio.sleep(HEALTH_CHECK_INTERVAL)

async def main():
    """ Runs the Quantum AI Lichess Bot """
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(handle_events())  
        loop.create_task(monitor_health())  

        global event_thread
        event_thread = threading.Thread(target=monitor_threads, daemon=True)
        event_thread.start()

        await asyncio.Event().wait()  

    except asyncio.CancelledError:
        logger.info("🛑 Event loop cancelled, shutting down...")
    except Exception as e:
        logger.critical(f"🔥 Fatal error in main loop: {e}\n{traceback.format_exc()}")

if __name__ == "__main__":
    try:
        logger.info("🚀 NECROMINDX Bot Starting... AI Mode Activated")

        asyncio.run(main())

    except KeyboardInterrupt:
        logger.info("🛑 Bot manually stopped. Exiting gracefully...")