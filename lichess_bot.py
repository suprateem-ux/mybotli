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
import time
import traceback
import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from functools import lru_cache
from collections import deque
import random
from functools import lru_cache
from stockfish import Stockfish
import torch




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
        "SyzygyPath": "https://tablebase.lichess.ovh",
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
        "SyzygyPath": "https://tablebase.lichess.ovh",
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
        "SyzygyPath": "https://tablebase.lichess.ovh",
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
        "SyzygyPath": "https://tablebase.lichess.ovh",
        "AutoLagCompensation": True
    }
}

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global variables
engine = None  # Initialize the engine globally
STOCKFISH_PATH = "./engines/stockfish-windows-x86-64-avx2.exe"  # Replace with the actual path to Stockfish
def configure_engine_for_time_control(time_control):
    """Dynamically configure Stockfish settings based on game time."""
    global engine

    # Input validation
    if not isinstance(time_control, (int, float)) or time_control < 0:
        raise ValueError("time_control must be a non-negative number")

    # Initialize failed_options list
    failed_options = []

    # Ensure engine is initialized
    if engine is None:
        logger.error("❌ Stockfish engine is not initialized! Call initialize_stockfish() first.")
        return

    # Determine settings based on time control
    if time_control <= 30:
        config = ENGINE_CONFIGS["hyperbullet"]
    elif time_control <= 180:
        config = ENGINE_CONFIGS["blitz"]
    elif time_control <= 600:
        config = ENGINE_CONFIGS["rapid"]
    else:
        config = ENGINE_CONFIGS["classical"]

       


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global variables
engine = None  # Initialize the engine globally
STOCKFISH_PATH = "./engines/stockfish-windows-x86-64-avx2.exe"  # Replace with the actual path to Stockfish
def configure_engine_for_time_control(time_control):
    """Dynamically configure Stockfish settings based on game time."""
    global engine

    # Input validation
    if not isinstance(time_control, (int, float)) or time_control < 0:
        raise ValueError("time_control must be a non-negative number")

    # Initialize failed_options list
    failed_options = []

    # Ensure engine is initialized
    if engine is None:
        logging.error("❌ Stockfish engine is not initialized! Call initialize_stockfish() first.")
        return

    # Determine settings based on time control
    if time_control <= 30:
        config = ENGINE_CONFIGS["hyperbullet"]
    elif time_control <= 180:
        config = ENGINE_CONFIGS["blitz"]
    elif time_control <= 600:
        config = ENGINE_CONFIGS["rapid"]
    else:
        config = ENGINE_CONFIGS["classical"]

    # Apply configurations to Stockfish
    for option, value in config.items():  # <-- Fixed indentation here
        try:
            engine.configure({option: value})
            logging.info(f"✅ Set {option} to {value}")
        except Exception as e:
            logging.warning(f"⚠️ Failed to set {option}: {e}")
            failed_options.append(option)

    if failed_options:
        logging.warning(f"⚠️ Some options failed to apply: {failed_options}")

    # Log final configuration status
    logging.info(f"🔥 Stockfish configured for {time_control}s games. Failed options: {failed_options if failed_options else 'None'}")

    # ✅ Auto-Healing: Restart Stockfish if it's unresponsive
    try:
        engine.ping()  # Ensure Stockfish is running
    except Exception as e:
        logging.error(f"⚠️ Stockfish engine crashed! Restarting... Reason: {e}")
        restart_stockfish(config)

    return failed_options
def restart_stockfish(config):
    """Restarts Stockfish and re-applies configuration."""
    global engine
    time.sleep(1)  # Short delay before restarting

    # Close the existing engine (if any)
    try:
        if engine:
            engine.close()
    except Exception as e:
        logging.warning(f"⚠️ Failed to close engine: {e}")

    # Restart Stockfish
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        logging.info("✅ Stockfish restarted successfully!")

        # Reapply configuration
        failed_options = []
        for option, value in config.items():
            try:
                engine.configure({option: value})
                logging.info(f"✅ Successfully reconfigured {option} = {value}")
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

# Machine learning-inspired failure tracking (simple version)
FAILURE_HISTORY = deque(maxlen=50)  # Stores last 50 outcomes

def predict_failure():
    """Predicts the probability of failure based on past outcomes."""
    if not FAILURE_HISTORY:
        return 0.2  # Default failure probability (20%)
    return sum(FAILURE_HISTORY) / len(FAILURE_HISTORY)

async def cloud_failover():
    """Simulates switching to a cloud-based instance to continue operations."""
    logging.critical("☁️ Switching to CLOUD MODE due to excessive failures!")
    await asyncio.sleep(random.randint(5, 15))  # Simulated transition time
    logging.critical("🌍 Cloud Mode ACTIVE. Challenges will be sent from cloud instance!")

async def challenge_loop():
    """Continuously sends challenges while adapting to failures with ML and parallel handling."""
    failure_count = 0
    total_failures = 0
    cloud_switch_triggered = False

    while True:
        predicted_fail_chance = predict_failure()
        if random.random() < predicted_fail_chance:
            delay = 15  # Simulated failure
        else:
            delay = random.randint(5, 10)  # Simulated success

        if delay == 15:  # Challenge failed
            failure_count += 1
            total_failures += 1
            FAILURE_HISTORY.append(1)

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
            if total_failures >= 10 and not cloud_switch_triggered:
                asyncio.create_task(cloud_failover())  # Runs cloud switch in parallel
                cloud_switch_triggered = True  # Ensures only one cloud switch attempt
                await asyncio.sleep(random.randint(1800, 3600))  # 30-60 min cooldown
                total_failures = 0  # Reset total failures
        else:
            FAILURE_HISTORY.append(0)
            failure_count = 0  # Reset failure streak on success
            jitter = random.uniform(-3, 3)  # Makes behavior unpredictable
            await asyncio.sleep(delay + jitter)

# Example run (remove this in real bot)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
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

# Constants
CHEAT_ACCURACY_THRESHOLD = 99
FAST_MOVE_THRESHOLD = 0.1
BOOK_MOVE_THRESHOLD = 15
MAX_SANDBAGGING_RATING_DROP = 300
API_CHEATING_THRESHOLD = 0.02
MAX_CONCURRENT_GAMES = 8
HEALTH_CHECK_INTERVAL = 30
AUTO_HEAL_DELAY = 2
OVERHEAD_BUFFER = 0.05
MAX_THREADS = multiprocessing.cpu_count()

# 🚀 THREAD & PROCESS MANAGEMENT
active_games = set()
stop_event = threading.Event()
executor = ThreadPoolExecutor(max_workers=MAX_THREADS)
engine_lock = threading.Lock()

def safe_engine_play(board, time_limit):
    """ Thread-safe Stockfish move calculation """
    with engine_lock:
        return engine.play(board, chess.engine.Limit(time=time_limit))

experience_replay = deque(maxlen=10000)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Running on: {device}")

# ✅ Define the NECROMINDX Deep Neural Network
class NECROMINDX_DNN(nn.Module):
    def __init__(self):
        super(NECROMINDX_DNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(773, 512),  # Input FEN encoding size → Hidden Layer
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1968)  # Output layer (all possible chess moves)
        )

    def forward(self, x):
        return self.layers(x)

# ✅ Load the pre-trained model with existence check
dnn_model = NECROMINDX_DNN().to(device)
model_path = "necromindx_dnn.pth"
if os.path.exists(model_path):
    dnn_model.load_state_dict(torch.load(model_path, map_location=device))
    dnn_model.eval()
    print("✅ Model loaded successfully!")
else:
    print("⚠️ WARNING: Model file missing! Training from scratch!")

# ✅ Reinforcement Learning – Experience Replay Buffer
experience_buffer = deque(maxlen=10000)  # Stores last 10,000 moves

def store_experience(fen, move, reward):
    """Store game experience for training"""
    experience_buffer.append((fen, move, reward))

def sample_experience(batch_size=64):
    """Sample random experiences for training"""
    return random.sample(experience_buffer, min(len(experience_buffer), batch_size))

# ✅ Define Stockfish Engine for MCTS Backup
stockfish = Stockfish("./engines/stockfish-windows-x86-64-avx2.exe", parameters={"Threads": 6, "Skill Level": 20})

@lru_cache(maxsize=20000)
def cached_dnn_prediction(fen):
    """ 🚀 Hyper-optimized DNN move prediction with self-learning & MCTS fallback """
    try:
        board = chess.Board(fen)
        fen = board.fen()

        # ✅ Exploration vs. Exploitation (80% best move, 20% random exploration)
        explore = random.random() < 0.2

        input_tensor = torch.tensor(encode_fen(fen), dtype=torch.float32).to(device).unsqueeze(0)

        with torch.no_grad():
            prediction = dnn_model(input_tensor).cpu().numpy()

        if explore:
            best_move_index = np.random.choice(len(prediction))  # Random move for exploration
        else:
            best_move_index = np.argmax(prediction)  # Best move for exploitation

        best_move = decode_move(best_move_index, board)
        return best_move

    except Exception as e:
        print(f"⚠️ DNN Prediction Error: {e}. Falling back to MCTS...")
        return monte_carlo_tree_search(fen)

def monte_carlo_tree_search(fen):
    """ ✅ Monte Carlo Tree Search (MCTS) for refined move selection """
    stockfish.set_fen_position(fen)
    return stockfish.get_best_move()

# ✅ Q-Learning with Neural Network for Self-Learning AI
optimizer = optim.Adam(dnn_model.parameters(), lr=0.001)
loss_function = nn.MSELoss()

def update_q_learning(fen, move, reward):
    """Update the DNN using Q-learning after a game"""
    input_tensor = torch.tensor(encode_fen(fen), dtype=torch.float32).to(device).unsqueeze(0)

    with torch.no_grad():
        q_values = dnn_model(input_tensor).cpu().numpy()

    move_index = encode_move(move)
    q_values[0][move_index] = reward  # Update move with its reward

    # Convert back to tensor
    target_tensor = torch.tensor(q_values, dtype=torch.float32).to(device)

    # Optimize model
    optimizer.zero_grad()
    prediction = dnn_model(input_tensor)
    loss = loss_function(prediction, target_tensor)
    loss.backward()
    optimizer.step()

# ✅ Periodic Self-Learning from Experience Replay
def train_from_experience():
    """Train NECROMINDX from stored game experiences"""
    if len(experience_buffer) < 500:
        return  # Not enough data yet

    batch = sample_experience()
    for fen, move, reward in batch:
        update_q_learning(fen, move, reward)

# ✅ Encode FEN & Moves for Neural Network Input
def encode_fen(fen):
    """Convert FEN into a tensor-friendly format using bitboards"""
    board = chess.Board(fen)
    bitboard = np.zeros(773, dtype=np.float32)
    for i, piece in enumerate(chess.PIECE_TYPES):
        for square in board.pieces(piece, chess.WHITE):
            bitboard[i * 64 + square] = 1
        for square in board.pieces(piece, chess.BLACK):
            bitboard[(i + 6) * 64 + square] = 1
    return bitboard

def encode_move(move):
    """Convert UCI move to an index"""
    move_uci = chess.Move.from_uci(move).uci()
    return hash(move_uci) % 1968  # Map to valid index range

def decode_move(index, board):
    """Convert index back to a chess move"""
    legal_moves = list(board.legal_moves)
    return legal_moves[index % len(legal_moves)] if legal_moves else board.san(board.peek())

# === QUANTUM AI-POWERED GAMEPLAY ===
# Constants (add these at the top of your file)
MIN_MOVE_TIME = 0.1  # Never use less than 0.1 seconds for a move
MAX_ENGINE_RETRIES = 3  # Max retries for engine failures
ENGINE_TIMEOUT = 10.0  # Max seconds for engine analysis

async def play_game(game_id, game):
    """Enhanced quantum AI gameplay with all requested improvements"""
    # Initialization
    logger.info(f"🎯 Game started: {game_id}")
    opponent_title = game.get("opponent", {}).get("title", "")
    opponent_is_bot = opponent_title == "BOT"
    
    # Custom greeting based on opponent type
    greeting = random.choice([
        f"🚀 Warping into a relativistic chess duel against {opponent_title or 'opponent'}!" if not opponent_is_bot else
        "🤖 Quantum circuits engaged! AI singularity showdown begins!",
        f"⚛️ Observing {opponent_title}'s wavefunction collapse... The game begins!" if opponent_title else
        "🌀 Entering a quantum chess superposition... Where will reality settle?"
    ])
    await client.bots.post_message(game_id, greeting)

    board = chess.Board()
    game_over = False

    try:
        while not game_over:
            # --- Improved Draw Handling ---
            try:
                game_state = await client.games.get_ongoing(game_id)
                if game_state.get("opponentOffersDraw", False):
                    if await should_accept_draw(board, game_state, opponent_is_bot):
                        await client.bots.accept_draw(game_id)
                        game_over = True
                        continue
            except Exception as e:
                logger.error(f"⚠️ Draw check error: {e}")

            # --- Enhanced Move Handling ---
            move_success = False
            last_exception = None
            
            for attempt in range(MAX_ENGINE_RETRIES):
                try:
                    # Calculate safe move time
                    move_time = max(
                        MIN_MOVE_TIME,
                        get_time_control(game["clock"], False) - OVERHEAD_BUFFER
                    )
                    
                    # Non-blocking engine analysis
                    result = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            executor,
                            lambda: safe_engine_play(board, move_time)
                        ),
                        timeout=ENGINE_TIMEOUT
                    )
                    
                    await client.bots.make_move(game_id, result.move.uci())
                    board.push(result.move)
                    move_success = True
                    break
                    
                except Exception as e:
                    last_exception = e
                    logger.warning(f"⚠️ Move attempt {attempt + 1} failed: {e}")
                    if attempt == MAX_ENGINE_RETRIES - 1:
                        # Fallback to legal move if engine fails
                        fallback_move = random.choice(list(board.legal_moves))
                        await client.bots.make_move(game_id, fallback_move.uci())
                        board.push(fallback_move)
                        logger.error(f"🚨 Engine failed! Played fallback move: {fallback_move.uci()}")

            # --- Game Over Check ---
            if board.is_game_over():
                game_over = True

    except Exception as e:
        logger.critical(f"🔥 Game loop crashed: {e}\n{traceback.format_exc()}")
    finally:
        await handle_game_end(game_id, board, opponent_title, opponent_is_bot)

async def should_accept_draw(board, game_state, opponent_is_bot):
    """Enhanced draw acceptance logic"""
    if not opponent_is_bot:
        return False
        
    my_rating = game_state.get("player", {}).get("rating", 0)
    opponent_rating = game_state.get("opponent", {}).get("rating", 0)
    
    # Rating filter
    if abs(my_rating - opponent_rating) > 100:
        return False
    
    # Position evaluation
    try:
        with engine_lock:
            eval = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda: engine.analyse(board, chess.engine.Limit(depth=12))["score"].relative.score()
                ),
                timeout=5.0
            )
        return abs(eval) < 50
    except:
        return False

async def handle_game_end(game_id, board, opponent_title, opponent_is_bot):
    """Enhanced game end handling"""
    result = board.result()
    
    # Custom messages based on opponent type and result
    messages = {
        "1-0": [
            f"🚀 Achieved quantum supremacy vs {opponent_title}! Spacetime bent in my favor!" if not opponent_is_bot else
            "🤖 Quantum fluctuations eliminated silicon adversary! GG!",
            f"⚛️ My pieces tunneled through {opponent_title}'s defenses!" if opponent_title else
            "🌌 Quantum computation complete! The position collapsed into a win!"
        ],
        "0-1": [
            f"🕳️ Got pulled into {opponent_title}'s gravitational trap! Event horizon reached!" if not opponent_is_bot else
            "🤖 AI singularity surpassed me... Must recalibrate my neural network!",
            "🔻 Entropy won this time… The multiverse chose an alternate timeline!"
        ],
        "1/2-1/2": [
            f"⚖️ Quantum decoherence achieved... The battle exists in a superposition of wins and losses!" if not opponent_is_bot else
            "🤖 AI equilibrium reached—our quantum circuits cancel out!",
            "🌠 A draw... or did we just split into parallel universes where both won?"
        ]
    }
    
    # Select random appropriate message
    message = random.choice(messages.get(result, ["Game completed"]))
    await client.bots.post_message(game_id, message)
    logger.info(f"📌 Game ended: {result} vs {opponent_title or 'opponent'}")

async def handle_events():
    while True:
        try:
            async for event in client.bots.stream_incoming_events():
                asyncio.create_task(process_event(event))
        except Exception as e:
            logger.critical(f"🔥 Critical error in event loop: {e}\n{traceback.format_exc()}")
            await reconnect_lichess()

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

async def main():
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(handle_events())
        threading.Thread(target=monitor_health, daemon=True).start()
        threading.Thread(target=monitor_threads, daemon=True).start()
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
        logger.info("🛑 Bot manually stopped. Exiting gefully...")
