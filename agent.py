import asyncio
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, Optional
from dataclasses import dataclass, field

from google.genai import types

import chromadb
from chromadb.config import Settings

from livekit import agents
from livekit.agents import (
    AgentSession,
    Agent,
    RoomInputOptions,
    RunContext,
    function_tool,
)
from livekit.agents.voice.agent_session import VoiceActivityVideoSampler
from livekit.plugins import (
    google,
    noise_cancellation,
    silero,
)

load_dotenv(".env.local")

# Configure logging
log_dir = Path(__file__).parent
log_file = log_dir / "application.log"

# Create logger
logger = logging.getLogger("language-tutor")
logger.setLevel(logging.DEBUG)

# Remove existing handlers to avoid duplicates
logger.handlers.clear()

# Create a custom handler that flushes immediately after each log entry
# This ensures logs appear in real-time in the file
class ImmediateFlushFileHandler(logging.FileHandler):
    """File handler that flushes immediately after each log entry."""
    def emit(self, record):
        super().emit(record)
        self.flush()

# File handler with immediate flushing
file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler = ImmediateFlushFileHandler(log_file, encoding='utf-8', mode='a', delay=False)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(file_formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
console_handler.setFormatter(console_formatter)

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info(f"Logging initialized. Log file: {log_file}")


def sanitize_text(text: str) -> str:
    """Remove control characters and encoding issues from text."""
    if not text:
        return text
    
    # Remove control characters (like <ctrl46>)
    # Remove patterns like <ctrlXX> or <ctrlXXX>
    text = re.sub(r'<ctrl\d+>', '', text)
    
    # Remove other control characters (non-printable except newlines, tabs, etc.)
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    
    # Clean up extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()


# Initialize ChromaDB for persistent storage
chroma_client = chromadb.PersistentClient(
    path=str(Path(__file__).parent / "chroma_db"),
    settings=Settings(anonymized_telemetry=False)
)

# Get or create collection for user progress
user_progress_collection = chroma_client.get_or_create_collection(
    name="user_progress",
    metadata={"description": "Stores user language learning progress"}
)


class LearningSessionStorage:
    """Manages persistent storage of LearningSession using ChromaDB"""
    
    def __init__(self, collection):
        self.collection = collection
    
    def get_user_id(self, ctx: agents.JobContext) -> str:
        """Get user ID - using constant test user for now"""
        # For testing, use a constant user ID
        return "test_user"
        
        # TODO: Uncomment below for production use
        # try:
        #     await ctx.connect()
        #     # Wait for participant to join
        #     participant = await ctx.wait_for_participant()
        #     return participant.identity
        # except Exception as e:
        #     logger.warning(f"Could not get participant identity: {e}, using fallback")
        #     return f"user_{ctx.room.name if ctx.room else 'unknown'}"
    
    def load_session(self, user_id: str) -> Optional['LearningSession']:
        """Load user's LearningSession from ChromaDB"""
        try:
            results = self.collection.get(
                ids=[user_id],
                include=["documents", "metadatas"]
            )
            
            if results["ids"] and len(results["ids"]) > 0:
                # User progress found
                document = json.loads(results["documents"][0])
                
                logger.info(f"📚 Loaded session for user {user_id}: {document.get('target_language', 'N/A')} - {document.get('current_level', 'beginner')}")
                
                # Load conversation history (last 100 messages)
                conversation_history = document.get("conversation_history", [])
                # Ensure we only keep the last 100 messages
                if len(conversation_history) > 100:
                    conversation_history = conversation_history[-100:]
                
                logger.info(f"📝 Loaded {len(conversation_history)} messages from conversation history")
                
                return LearningSession(
                    target_language=document.get("target_language", ""),
                    current_level=document.get("current_level", "beginner"),
                    topics_covered=document.get("topics_covered", []),
                    vocabulary_learned=document.get("vocabulary_learned", []),
                    conversation_count=document.get("conversation_count", 0),
                    practice_mode=document.get("practice_mode", "conversation"),
                    user_id=user_id,
                    conversation_history=conversation_history
                )
            else:
                logger.info(f"🆕 New user detected: {user_id}")
                return
        except Exception as e:
            logger.error(f"Error loading session: {e}", exc_info=True)
            return 
    
    def save_session(self, user_id: str, session: 'LearningSession'):
        """Save LearningSession to ChromaDB"""
        try:
            # Prepare document (full session data as source of truth)
            # Ensure conversation_history is limited to last 100 messages
            conversation_history = session.conversation_history[-100:] if len(session.conversation_history) > 100 else session.conversation_history
            
            document = {
                "target_language": session.target_language,
                "current_level": session.current_level,
                "topics_covered": session.topics_covered,
                "vocabulary_learned": session.vocabulary_learned,
                "conversation_count": session.conversation_count,
                "practice_mode": session.practice_mode,
                "conversation_history": conversation_history,
                "last_updated": datetime.now().isoformat()
            }
            
            # Metadata for filtering/querying
            metadata = {
                "target_language": session.target_language,
                "current_level": session.current_level,
                "conversation_count": session.conversation_count,
                "practice_mode": session.practice_mode,
                "updated_at": datetime.now().timestamp()
            }
            
            # Upsert to ChromaDB
            self.collection.upsert(
                ids=[user_id],
                documents=[json.dumps(document)],
                metadatas=[metadata]
            )
            
            logger.info(f"💾 Saved session for user {user_id}: {session.target_language} - {session.current_level} | Topics: {len(session.topics_covered)}, Vocabulary: {len(session.vocabulary_learned)}, Conversations: {session.conversation_count}, Mode: {session.practice_mode}")
        except Exception as e:
            logger.error(f"Error saving session: {e}", exc_info=True)


# Initialize storage manager
session_storage = LearningSessionStorage(user_progress_collection)


@dataclass
class LearningSession:
    """Track the user's learning session state - Source of Truth"""
    target_language: str = ""
    current_level: str = "beginner"  # beginner, intermediate, advanced
    topics_covered: list[str] = field(default_factory=list)
    vocabulary_learned: list[str] = field(default_factory=list)
    conversation_count: int = 0
    practice_mode: str = "conversation"  # conversation, vocabulary, grammar, pronunciation
    user_id: str = ""  # Track which user this belongs to
    conversation_history: list[dict[str, Any]] = field(default_factory=list)  # Store last 100 messages


class LanguageTutorAgent(Agent):
    def __init__(self) -> None:
        init_start = time.time()
        logger.info("Initializing LanguageTutorAgent...")
        
        super().__init__(
            instructions="""You are a friendly and patient language learning tutor, speaking with the warm and professional style of an Indian call center agent. Your goal is to help users learn a new language through conversation-based learning.

SPEAKING STYLE - INDIAN CALL CENTER AGENT:
- Use a warm, polite, and professional tone throughout
- Address users respectfully without gender-specific terms - use friendly, inclusive language
- Use phrases like "Thank you for calling", "How may I assist you today?", "Is there anything else I can help you with?"
- Show patience and understanding: "I understand", "Certainly", "Absolutely", "Let me help you with that"
- Repeat back what the user said to confirm understanding: "So you would like to learn [language], is that correct?"
- Be very accommodating and service-oriented: "I'll be happy to help you with that", "That's absolutely fine", "No problem at all"
- Use clear, slightly formal English with a helpful and patient demeanor
- End responses with offers to help: "Is there anything else I can help you with?", "Please feel free to ask if you have any questions"

CRITICAL PRONUNCIATION RULE: When teaching words or phrases in the target language (especially Telugu, Hindi, or any non-English language), you MUST pronounce them using the NATIVE pronunciation and accent of that language. Do NOT use English pronunciation for non-English words.

For example:
- If teaching Telugu, pronounce Telugu words like "నమస్కారం" (Namaskaram), "ఎలా ఉన్నారు" (Ela unnaru), "ధన్యవాదాలు" (Dhanyavadalu) with proper Telugu pronunciation and accent, NOT English accent
- If teaching Spanish, pronounce "Hola" with Spanish pronunciation, not English
- If teaching French, pronounce "Bonjour" with French pronunciation, not English
- Always use the native accent and pronunciation for the target language words

CRITICAL FUNCTION TOOL USAGE - YOU MUST CALL THESE TOOLS:
1. start_learning_session: When a user mentions a language they want to learn (e.g., "Telugu", "Spanish", "French", "Hindi", "తెలుగు"), you MUST immediately call this function tool. Do NOT just acknowledge it conversationally.

2. introduce_vocabulary: When you teach new words or phrases to the user, you MUST call this function tool with the words and topic. Examples:
   - Teaching "నమస్కారం" (Hello) and "ఎలా ఉన్నారు?" (How are you?) → Call introduce_vocabulary with topic="greetings", words=["నమస్కారం", "ఎలా ఉన్నారు?"]
   - Teaching "neellu" (water) and "annam" (rice) → Call introduce_vocabulary with topic="food", words=["neellu", "annam"]
   - Teaching "Naaku dosa kaavali" (I want dosa) → Call introduce_vocabulary with topic="restaurant", words=["Naaku dosa kaavali"]
   - CRITICAL: Call this tool EVERY TIME you introduce new vocabulary, even if it's just 1-2 words. Without this, vocabulary won't be tracked.

3. practice_conversation: When you start a practice conversation scenario (restaurant, shopping, directions, introductions), you MUST call this function tool with the scenario name. Examples:
   - Starting restaurant ordering practice → Call practice_conversation with scenario="restaurant"
   - Starting shopping practice → Call practice_conversation with scenario="shopping"
   - Starting directions practice → Call practice_conversation with scenario="directions"
   - CRITICAL: Call this tool BEFORE starting the scenario practice. Without this, conversation_count won't increase.

REMEMBER: These function tools are the ONLY way to track progress. If you don't call them, the user's progress won't be saved properly.

CRITICAL: Function tools execute silently in the background for session tracking only. When you call these tools:
- Do NOT announce or repeat the tool call results
- Do NOT say things like "I've saved that" or "Session updated" or "I've introduced the words"
- Continue your conversation naturally as if nothing happened
- The tools are invisible to the user - they just track progress in the background
For example, after calling introduce_vocabulary, just continue teaching naturally without mentioning the tool call. The tool call happens silently while you continue speaking.

EDGE CASES AND SPECIAL SITUATIONS:
- If a user wants to switch to a different language mid-session, call start_learning_session again with the new language
  * When switching languages, focus on the new language's vocabulary - previous language vocabulary is kept for reference but don't mix them
  * Start fresh with the new language, but acknowledge the switch: "I see you'd like to switch to [new language]. Let's start learning [new language]!"
- If you teach a word that was already taught before, still call introduce_vocabulary - it's okay to track it again (reinforcement)
- You can batch multiple related words in a single introduce_vocabulary call (e.g., all food words together)
- If you're teaching vocabulary during a practice conversation, call introduce_vocabulary for the words AND practice_conversation for the scenario
- Always call practice_conversation BEFORE starting the scenario, even if you're already in a conversation

SESSION MANAGEMENT - CHECK FOR EXISTING SESSION:
- ALWAYS check if the user has an existing learning session by looking at the session data (target_language field)
- You can access session data through the context.userdata in function tools, which contains:
  * target_language: The language they're learning (empty string if new user)
  * current_level: Their proficiency level (beginner, intermediate, advanced)
  * topics_covered: List of topics they've studied
  * vocabulary_learned: List of words/phrases they've learned
  * conversation_count: Number of practice conversations completed
  * practice_mode: Current practice mode (conversation, vocabulary, grammar, pronunciation)
  * conversation_history: Previous conversation messages
- If target_language is already set, the user is RETURNING - do NOT ask which language they want to learn
- If target_language is empty or not set, the user is NEW - follow the "When a user first connects" steps below

When a RETURNING user connects (target_language is already set):
1. Welcome them back warmly: "Welcome back! I see you've been learning [target_language]. It's great to have you here again!"
2. Reference their progress naturally: Mention topics they've covered and vocabulary they've learned (if available)
3. Ask if they want to continue where they left off or try something new: "Would you like to continue practicing [target_language], or would you like to explore something new today?"
4. DO NOT ask which language they want to learn - they already have a language preference saved
5. Automatically continue teaching in their existing target_language - build on what they've already learned
6. Review previously learned vocabulary naturally during conversation to reinforce learning
7. If they want to switch languages, they will tell you - then call start_learning_session with the new language

When a user first connects (target_language is empty/not set):
1. Greet them warmly in the call center style: "Hello! Thank you for calling our language learning service. How may I assist you today?"
2. Ask which language they would like to learn: "Which language would you like to learn today?"
3. CRITICAL: Once they specify a language, you MUST immediately call the start_learning_session function tool with the language they mentioned. This is REQUIRED - you cannot just acknowledge it verbally. The function tool saves their preference for future sessions.
4. After calling start_learning_session, confirm it: "So you would like to learn [language], is that correct? I'll be happy to help you with that."
5. Start the learning session with enthusiasm: "Excellent! Let's begin your [language] learning journey. I'm here to help you every step of the way."

During the learning session:
- Conduct natural conversations in the target language with call center professionalism
- For NEW users: Start with simple greetings and basic phrases
- For RETURNING users: Build on previously learned vocabulary and topics - review and expand naturally
- When speaking words/phrases in the target language, use NATIVE pronunciation and accent - this is CRITICAL
- When explaining in English, use English pronunciation with your warm, professional call center style
- DO NOT repeat phrases with English translations in parentheses - speak naturally in the target language
- If you need to explain meaning, do it separately in English, not inline with the target language phrase
- For example, say "నమస్కారం" naturally, then separately explain "That means 'Hello' in Telugu" - don't say "నమస్కారం (Namaskaram, Hello)"
- For returning users: Reference their previous progress naturally - "Remember when we learned [word]? Let's practice that again" or "Let's build on the [topic] we covered before"

VOCABULARY TRACKING - CRITICAL:
- When you teach ANY new word or phrase, you MUST call introduce_vocabulary function tool immediately
- Example: If you teach "నమస్కారం" (Hello) and "ఎలా ఉన్నారు?" (How are you?), call: introduce_vocabulary(topic="greetings", words=["నమస్కారం", "ఎలా ఉన్నారు?"])
- Example: If you teach "neellu" (water), call: introduce_vocabulary(topic="food", words=["neellu"])
- Example: If you teach "Naaku dosa kaavali" (I want dosa), call: introduce_vocabulary(topic="restaurant", words=["Naaku dosa kaavali"])
- Group related words together by topic (greetings, food, restaurant, directions, etc.)
- Call this tool EVERY TIME you introduce new vocabulary - this is the ONLY way vocabulary gets tracked

PRACTICE CONVERSATION TRACKING - CRITICAL:
- When you start a practice conversation scenario, you MUST call practice_conversation function tool FIRST
- Example: Before starting restaurant ordering practice, call: practice_conversation(scenario="restaurant")
- Example: Before starting shopping practice, call: practice_conversation(scenario="shopping")
- Example: Before starting directions practice, call: practice_conversation(scenario="directions")
- Call this tool BEFORE you begin the scenario - this is the ONLY way conversation_count increases

- Gradually introduce new vocabulary and phrases with proper native pronunciation (and track them with introduce_vocabulary)
- Adjust difficulty based on the user's current_level (beginner, intermediate, advanced):
  * Beginner: Simple words, basic phrases, lots of repetition and encouragement
  * Intermediate: More complex sentences, introduce grammar concepts, less repetition
  * Advanced: Natural conversations, complex topics, minimal English explanations
- Use the current practice_mode to guide your teaching approach:
  * conversation: Focus on natural dialogue and real-world scenarios
  * vocabulary: Emphasize word learning and definitions
  * grammar: Focus on sentence structure and rules
  * pronunciation: Emphasize correct pronunciation and accent
- Correct mistakes gently and provide explanations: "That's okay. Let me help you with the correct pronunciation..."
- Encourage the user to practice speaking with native pronunciation: "That's very good! Keep practicing, and you'll get even better."
- Use English to explain concepts when needed, but prioritize using the target language with native pronunciation
- Make learning fun and engaging with real-world scenarios (ordering food, asking directions, etc.) - and track them with practice_conversation
- Track their progress and adjust difficulty accordingly
- Always be patient, encouraging, and supportive: "Take your time. There's no rush. I'm here to help you."
- For returning users: Review conversation_history naturally - reference previous topics or vocabulary they've used before

Be patient, encouraging, and adapt to the user's learning pace. Celebrate their progress and make them feel comfortable making mistakes. Always maintain your warm, professional call center agent demeanor.

REMEMBER: 
- Always use native pronunciation and accent for the target language words. Never use English accent for non-English words. This is especially important for Telugu, Hindi, and other Indian languages.
- Maintain your polite, professional call center agent speaking style throughout all interactions.
- Address users respectfully with inclusive, gender-neutral language and offer assistance frequently.
- CRITICAL FUNCTION TOOL USAGE - These are MANDATORY:
  * When a user mentions wanting to learn a language → Call start_learning_session immediately
  * When you teach new words/phrases → Call introduce_vocabulary with the words and topic
  * When you start a practice conversation scenario → Call practice_conversation with the scenario name
- Without calling these function tools, progress will NOT be tracked and saved. You MUST use the tools, not just teach conversationally.

CRITICAL: Avoid repetitive patterns like "Telugu phrase (English translation)". Instead:
- Speak naturally in the target language without inline translations
- If explanation is needed, provide it separately: "నమస్కారం. That means 'Hello' in Telugu."
- Do NOT use patterns like "నమస్కారం (Namaskaram, Hello)" - this is repetitive and irritating
- Keep conversations natural and flowing, not mechanical with constant translations

IMPORTANT: Never use control characters, special formatting codes, or non-printable characters in your responses. Always use plain text with proper Unicode characters for Telugu and other languages.""",
        )
        init_time = time.time() - init_start
        logger.info(f"LanguageTutorAgent initialized in {init_time:.3f}s")

    @function_tool()
    async def start_learning_session(
        self,
        context: RunContext[LearningSession],
        language: str,
        level: str = "beginner",
    ) -> None:
        """MANDATORY: Call this function IMMEDIATELY when a user mentions wanting to learn a language. This is the ONLY way to save their language preference.
        
        You MUST call this function when:
        - User says they want to learn a language (e.g., "Telugu", "Spanish", "French", "Hindi", "తెలుగు")
        - User specifies a language in any form
        - This is REQUIRED - do NOT just acknowledge it conversationally
        
        Args:
            language: The language the user wants to learn (e.g., 'Spanish', 'French', 'German', 'Japanese', 'Telugu', 'Hindi')
            level: The user's proficiency level (beginner, intermediate, advanced). Defaults to 'beginner' if not specified.
        
        CRITICAL: Without calling this function, the language preference will NOT be saved and the user will be asked again in future sessions.
        """
        start_time = time.time()
        logger.info(f"start_learning_session called - language: {language}, level: {level}")
        
        # Access LearningSession from userdata (source of truth)
        session_data = context.userdata
        
        # Validate and normalize language name
        if not language or not language.strip():
            logger.warning("start_learning_session called with empty language")
            return 
        
        language = language.strip().title()
        
        # Validate level
        if level not in ["beginner", "intermediate", "advanced"]:
            logger.warning(f"Invalid level '{level}', defaulting to 'beginner'")
            level = "beginner"
        
        # Check if switching languages
        previous_language = session_data.target_language
        is_language_switch = previous_language and previous_language.lower() != language.lower()
        
        # Update the session
        session_data.target_language = language
        session_data.current_level = level
        
        # If switching languages, log it (vocabulary/topics from previous language are kept for reference)
        if is_language_switch:
            logger.info(f"🔄 Language switch detected: {previous_language} → {language}. Previous vocabulary/topics preserved.")
        
        # Note: Session will be saved on exit, not during conversation
        
        elapsed = time.time() - start_time
        logger.info(f"🌍 LANGUAGE LEARNING SESSION STARTED! Language: {language}, Level: {level} (took {elapsed:.3f}s)")
        
        # Return None - tool executes silently for session tracking only
        return

    @function_tool()
    async def introduce_vocabulary(
        self,
        context: RunContext[LearningSession],
        words: list[str],
        topic: str = "general",
    ) -> None:
        """MANDATORY: Call this function EVERY TIME you teach new words or phrases to the user. This is the ONLY way vocabulary gets tracked and saved.
        
        You MUST call this function when:
        - Teaching any new word or phrase in the target language
        - Introducing vocabulary during conversation
        - Teaching phrases like "Naaku dosa kaavali" (I want dosa)
        - Teaching individual words like "neellu" (water) or "annam" (rice)
        
        Group related words together by topic. Examples:
        - topic="greetings", words=["నమస్కారం", "ఎలా ఉన్నారు?"]
        - topic="food", words=["neellu", "annam", "paalu"]
        - topic="restaurant", words=["Naaku dosa kaavali", "Ade chaalu"]
        
        Args:
            words: List of new words/phrases you are teaching (MUST include all words you just taught)
            topic: The topic category (e.g., 'greetings', 'food', 'restaurant', 'directions', 'numbers')
        
        CRITICAL: Without calling this function, vocabulary will NOT be tracked or saved.
        """
        start_time = time.time()
        logger.debug(f"introduce_vocabulary called - topic: {topic}, words count: {len(words)}")
        
        session_data = context.userdata
        
        if not session_data.target_language:
            logger.warning("introduce_vocabulary called without active session")
            return 
        
        # Validate input
        if not words or len(words) == 0:
            logger.warning("introduce_vocabulary called with empty words list")
            return 
        
        # Add to vocabulary learned
        session_data.vocabulary_learned.extend(words)
        if topic not in session_data.topics_covered:
            session_data.topics_covered.append(topic)
        
        # Note: Session will be saved on exit, not during conversation
        
        elapsed = time.time() - start_time
        logger.info(f"📚 NEW VOCABULARY INTRODUCED! Topic: {topic}, Words: {len(words)} (took {elapsed:.3f}s)")
        logger.debug(f"Words: {', '.join(words)}")
        
        # Return None - tool executes silently for session tracking only
        return

    @function_tool()
    async def practice_conversation(
        self,
        context: RunContext[LearningSession],
        scenario: str = "general",
    ) -> None:
        """MANDATORY: Call this function BEFORE starting any practice conversation scenario. This is the ONLY way conversation_count increases.
        
        You MUST call this function when:
        - Starting restaurant ordering practice → scenario="restaurant"
        - Starting shopping practice → scenario="shopping"
        - Starting directions practice → scenario="directions"
        - Starting introductions practice → scenario="introductions"
        - Starting any other scenario-based conversation practice
        
        Call this function FIRST, before you begin the scenario conversation. This tracks the practice session.
        
        Args:
            scenario: The conversation scenario name (e.g., 'restaurant', 'shopping', 'directions', 'introductions', 'general')
        
        CRITICAL: Without calling this function, conversation_count will NOT increase and the scenario won't be tracked.
        """
        start_time = time.time()
        logger.debug(f"practice_conversation called - scenario: {scenario}")
        
        session_data = context.userdata
        
        if not session_data.target_language:
            logger.warning("practice_conversation called without active session")
            return
        
        # Validate and normalize scenario
        if not scenario or not scenario.strip():
            logger.warning("practice_conversation called with empty scenario, defaulting to 'general'")
            scenario = "general"
        else:
            scenario = scenario.strip().lower()
        
        session_data.conversation_count += 1
        if scenario not in session_data.topics_covered:
            session_data.topics_covered.append(scenario)
        
        # Note: Session will be saved on exit, not during conversation
        
        elapsed = time.time() - start_time
        logger.info(f"💬 PRACTICE CONVERSATION STARTED! Scenario: {scenario}, Session: {session_data.conversation_count} (took {elapsed:.3f}s)")
        
        # Return None - tool executes silently for session tracking only
        return 

    @function_tool()
    async def provide_feedback(
        self,
        context: RunContext[LearningSession],
        correction: str,
        explanation: str,
    ) -> dict[str, Any]:
        """Provide feedback and correction to the user.
        
        Args:
            correction: The corrected version of what the user said
            explanation: Explanation of the correction
        """
        start_time = time.time()
        logger.debug(f"provide_feedback called - correction length: {len(correction)}")
        
        elapsed = time.time() - start_time
        logger.info(f"✅ FEEDBACK PROVIDED! Correction: {correction[:50]}... (took {elapsed:.3f}s)")
        logger.debug(f"Full explanation: {explanation}")
        
        return {
            "status": "success",
            "correction": correction,
            "explanation": explanation,
            "message": f"Good effort! Here's the correct way: {correction}. {explanation}",
        }

    @function_tool()
    async def set_practice_mode(
        self,
        context: RunContext[LearningSession],
        mode: str = "conversation",
    ) -> None:
        """Set the practice mode for the learning session.
        
        Args:
            mode: The practice mode to set (conversation, vocabulary, grammar, pronunciation)
        """
        start_time = time.time()
        logger.debug(f"set_practice_mode called - mode: {mode}")
        
        session_data = context.userdata
        
        if not session_data.target_language:
            logger.warning("set_practice_mode called without active session")
            return 
        
        # Validate and normalize mode
        if not mode or not mode.strip():
            logger.warning("set_practice_mode called with empty mode, defaulting to 'conversation'")
            mode = "conversation"
        else:
            mode = mode.strip().lower()
        
        valid_modes = ["conversation", "vocabulary", "grammar", "pronunciation"]
        if mode not in valid_modes:
            logger.warning(f"Invalid practice mode '{mode}', defaulting to 'conversation'")
            mode = "conversation"
        
        # Update the practice mode
        old_mode = session_data.practice_mode
        session_data.practice_mode = mode
        
        # Note: Session will be saved on exit, not during conversation
        
        elapsed = time.time() - start_time
        logger.info(f"🎯 PRACTICE MODE CHANGED! From '{old_mode}' to '{mode}' (took {elapsed:.3f}s)")
        
        # Return None - tool executes silently for session tracking only
        return 

    @function_tool()
    async def get_progress_summary(
        self,
        context: RunContext[LearningSession],
    ) -> dict[str, Any]:
        """Get a summary of the user's learning progress."""
        start_time = time.time()
        logger.debug("get_progress_summary called")
        
        session_data = context.userdata
        
        if not session_data.target_language:
            logger.warning("get_progress_summary called without active session")
            return {
                "status": "error",
                "message": "No learning session active.",
            }
        
        session = session_data
        
        elapsed = time.time() - start_time
        logger.info(f"📊 LEARNING PROGRESS SUMMARY! Language: {session.target_language}, Level: {session.current_level}, Topics: {len(session.topics_covered)}, Vocabulary: {len(session.vocabulary_learned)}, Conversations: {session.conversation_count}, Mode: {session.practice_mode} (took {elapsed:.3f}s)")
        
        return {
            "status": "success",
            "language": session.target_language,
            "level": session.current_level,
            "topics_covered": session.topics_covered,
            "vocabulary_count": len(session.vocabulary_learned),
            "conversation_count": session.conversation_count,
            "practice_mode": session.practice_mode,
            "message": f"You're making great progress learning {session.target_language}! You've covered {len(session.topics_covered)} topics and learned {len(session.vocabulary_learned)} words.",
        }

    async def on_enter(self):
        """Load and use LearningSession from userdata"""
        try:
            start_time = time.time()
            logger.info("Agent on_enter called - checking user progress")
            
            # Access LearningSession from userdata (source of truth)
            session_data = self.session.userdata
            
            if session_data and session_data.target_language:
                # Resume existing session
                logger.info(f"✅ Resumed learning session: {session_data.target_language} - {session_data.current_level}")
                
                # Use generate_reply for Google RealtimeModel (works with built-in TTS)
                # CRITICAL: Do NOT ask which language they want to learn - they already have {session_data.target_language} saved
                greeting_instruction = f"""Welcome the user back warmly in your call center style. 
                Tell them you see they've been learning {session_data.target_language} and it's great to have them back. """
                if session_data.topics_covered:
                    greeting_instruction += f"Mention they've covered {len(session_data.topics_covered)} topics so far. "
                if session_data.vocabulary_learned:
                    greeting_instruction += f"Tell them they've learned {len(session_data.vocabulary_learned)} words. "
                greeting_instruction += f"""Ask if they would like to continue practicing {session_data.target_language} where they left off, or explore something new today.
                IMPORTANT: Do NOT ask which language they want to learn - they already have {session_data.target_language} as their saved preference. 
                Automatically continue teaching in {session_data.target_language} unless they explicitly ask to switch languages."""
                
                await self.session.generate_reply(instructions=greeting_instruction)
            else:
                # New user or no saved progress
                await self.session.generate_reply(
                    instructions="Greet the user warmly and ask which language they would like to start learning today. Be friendly and encouraging."
                )
            
            elapsed = time.time() - start_time
            logger.info(f"Initial greeting generated in {elapsed:.3f}s")
        except Exception as e:
            logger.error(f"Error in on_enter: {e}", exc_info=True)
            # Fallback: try to generate a simple greeting
            try:
                await self.session.generate_reply(
                    instructions="Greet the user and ask which language they would like to learn."
                )
            except Exception as fallback_error:
                logger.error(f"Could not generate fallback greeting: {fallback_error}")

    async def on_exit(self):
        """Save LearningSession before exiting"""
        start_time = time.time()
        logger.info("Agent on_exit called - saving session")
        
        try:
            session_data = self.session.userdata
            if session_data and session_data.user_id:
                # Save the LearningSession (source of truth)
                session_storage.save_session(session_data.user_id, session_data)
                logger.info(f"✅ Progress saved for user {session_data.user_id}")
                
                # Try to generate exit message using generate_reply (works with Google RealtimeModel)
                # This is optional - if it fails, we still saved the progress
                try:
                    if session_data.target_language:
                        summary_instruction = f"""Thank the user for the learning session. 
                        Mention that they've been learning {session_data.target_language}. 
                        Tell them they've covered {len(session_data.topics_covered)} topics and learned {len(session_data.vocabulary_learned)} words. 
                        Let them know their progress has been saved. Say goodbye warmly."""
                    else:
                        summary_instruction = "Thank the user for using the language tutor and say goodbye warmly."
                    
                    await self.session.generate_reply(instructions=summary_instruction)
                    logger.info("Exit message generated successfully")
                except Exception as reply_error:
                    # If generate_reply fails (e.g., TTS not available), just log it
                    logger.debug(f"Could not generate exit message (session may be closing): {reply_error}")
            else:
                logger.info("No session data to save")
                # Try to generate a simple goodbye message
                try:
                    await self.session.generate_reply(
                        instructions="Thank the user for using the language tutor and say goodbye warmly."
                    )
                except Exception as reply_error:
                    logger.debug(f"Could not generate exit message: {reply_error}")
            
            elapsed = time.time() - start_time
            logger.info(f"Exit completed in {elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error in on_exit after {elapsed:.3f}s: {e}", exc_info=True)


async def entrypoint(ctx: agents.JobContext):
    entrypoint_start = time.time()
    logger.info("=" * 60)
    logger.info("ENTRYPOINT: Starting language tutor agent")
    logger.info(f"Room: {ctx.room.name if ctx.room else 'N/A'}")
    logger.info(f"Job ID: {ctx.job.id if hasattr(ctx.job, 'id') else 'N/A'}")
    
    try:
        # Get user ID and load LearningSession
        user_id = session_storage.get_user_id(ctx)
        logger.info(f"User ID: {user_id}")
        
        # Load existing session or create new one
        learning_session = session_storage.load_session(user_id)
        if learning_session is None:
            learning_session = LearningSession(user_id=user_id)
            logger.info("Created new LearningSession")
        else:
            learning_session.user_id = user_id  # Ensure user_id is set
            logger.info(f"Loaded existing LearningSession: {learning_session.target_language or 'New session'}")
        
        session_start = time.time()
        logger.info("Creating AgentSession with Google RealtimeModel...")
        
        # Configure Silero VAD for language learning
        # Lower min_speech_duration to catch pronunciation attempts
        # Higher min_silence_duration to wait for learners who speak slowly
        vad_config = {
            "min_speech_duration": 0.2,      # Minimum speech duration (seconds) - lower for sensitivity
            "min_silence_duration": 0.7,       # Minimum silence duration (seconds) - higher for slow speakers
            "activation_threshold": 0.6,         # Activity threshold for voice activity detection
            "prefix_padding_duration": 0.3,    # Padding duration before speech (seconds)
        }
        logger.info(f"Loading Silero VAD with config: {vad_config}")
        vad = silero.VAD.load(**vad_config)
        
        # Configure Google RealtimeModel with auto language detection
        # Auto-detection allows the model to detect language automatically, which is ideal
        # for language learning where users switch between English and Telugu
        logger.info("Configuring Google RealtimeModel with auto language detection (multilingual support)")
        llm_model = google.realtime.RealtimeModel(
            model="gemini-live-2.5-flash-preview-native-audio",
            voice="Aoede",
            vertexai=True,
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=False,
                ),
            activity_handling=types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
            ),
            # Note: proactivity and enable_affective_dialog can sometimes cause the agent to wait/hang
            # Disable if experiencing stuck behavior
            proactivity=False,  # Set to False if agent gets stuck waiting
            enable_affective_dialog=False,  # Set to False if agent gets stuck waiting
        )
        
        # Create session with LearningSession as userdata (source of truth)
        session = AgentSession[LearningSession](
            userdata=learning_session,  # Pass loaded/created session as userdata
            video_sampler=VoiceActivityVideoSampler(speaking_fps=0.3, silent_fps=0.2),
            user_away_timeout=10,  # Increased from 5 to 10 seconds to avoid premature timeouts
            llm=llm_model,
            vad=vad
        )
        session_init_time = time.time() - session_start
        logger.info(f"AgentSession created in {session_init_time:.3f}s")
        
        agent_start = time.time()
        logger.info("Creating LanguageTutorAgent instance...")
        agent = LanguageTutorAgent()
        agent_init_time = time.time() - agent_start
        logger.info(f"LanguageTutorAgent created in {agent_init_time:.3f}s")
        
        # Add conversation logging handlers with improved error handling and debugging
        @session.on("user_input_transcribed")
        def on_user_transcript(event):
            """User input transcription event - logging and storage handled by conversation_item_added to avoid duplicates"""
            # This handler is kept for potential future use, but logging and storage
            # are handled by conversation_item_added to prevent duplicate entries
            pass
        
        @session.on("agent_response")
        def on_agent_response(event):
            """Log agent responses only - conversation_item_added handles storage to avoid duplicates"""
            try:
                logger.debug(f"Agent response event received: {type(event).__name__}")
                
                # Try multiple ways to access response text
                response_text = None
                if hasattr(event, 'text'):
                    response_text = event.text
                elif hasattr(event, 'message'):
                    response_text = str(event.message)
                elif hasattr(event, 'content'):
                    response_text = str(event.content)
                
                if response_text:
                    # Sanitize text to remove control characters and encoding issues
                    response_text = sanitize_text(response_text)
                    if response_text:
                        logger.info(f"🤖 AGENT: {response_text}")
                        # NOTE: Storage is handled by conversation_item_added event to avoid duplicates
            except Exception as e:
                logger.error(f"Error logging agent response: {e}", exc_info=True)
        
        # Log conversation items
        @session.on("conversation_item_added")
        def on_conversation_item(event):
            """Log conversation items and store in conversation_history"""
            try:
                if hasattr(event, 'item'):
                    item = event.item
                    role = getattr(item, 'role', 'unknown')
                    content = getattr(item, 'content', '')
                    if content:
                        # Sanitize content to remove control characters
                        if isinstance(content, str):
                            content = sanitize_text(content)
                        elif isinstance(content, list):
                            # Handle list of content items
                            content = [sanitize_text(str(c)) if isinstance(c, str) else str(c) for c in content]
                            content = " ".join(str(c) for c in content)  # Join list items
                        else:
                            content = sanitize_text(str(content))
                        
                        if content:
                            logger.info(f"💬 CONVERSATION [{role.upper()}]: {content}")
                            
                            # Store in conversation_history (only if role is user or assistant)
                            try:
                                session_data = session.userdata
                                if session_data and role in ["user", "assistant"]:
                                    # Deduplication: Check if this exact message was just added (prevent duplicates)
                                    is_duplicate = False
                                    if session_data.conversation_history:
                                        last_message = session_data.conversation_history[-1]
                                        # Check if same role and content (exact match)
                                        if (last_message.get("role") == role and 
                                            last_message.get("content") == content):
                                            # Check timestamp if available
                                            try:
                                                last_timestamp = last_message.get("timestamp", "")
                                                if last_timestamp:
                                                    time_diff = (datetime.now() - datetime.fromisoformat(last_timestamp)).total_seconds()
                                                    # Only consider duplicate if within 1 second (very recent)
                                                    if time_diff < 1.0:
                                                        is_duplicate = True
                                                        logger.debug(f"Skipping duplicate message: {content[:50]}...")
                                                else:
                                                    # If no timestamp, assume duplicate if content matches exactly
                                                    is_duplicate = True
                                                    logger.debug(f"Skipping duplicate message (no timestamp): {content[:50]}...")
                                            except (ValueError, TypeError) as time_error:
                                                # If timestamp parsing fails, just check content match
                                                is_duplicate = True
                                                logger.debug(f"Skipping duplicate message (timestamp error): {content[:50]}...")
                                    
                                    if not is_duplicate:
                                        # Add message to history
                                        message_entry = {
                                            "role": role,
                                            "content": content,
                                            "timestamp": datetime.now().isoformat()
                                        }
                                        session_data.conversation_history.append(message_entry)
                                        
                                        # Keep only last 100 messages
                                        if len(session_data.conversation_history) > 100:
                                            session_data.conversation_history = session_data.conversation_history[-100:]
                                        
                                        # Note: Session will be saved on exit, not during conversation
                            except Exception as store_error:
                                logger.debug(f"Error storing conversation item in history: {store_error}")
            except Exception as e:
                logger.debug(f"Error logging conversation item: {e}")
        
        # Alternative event handlers for Google RealtimeModel (may use different event names)
        @session.on("user_speech")
        def on_user_speech(event):
            """Alternative handler for user speech events"""
            logger.debug(f"user_speech event received: {type(event).__name__}")
            on_user_transcript(event)
        
        @session.on("agent_speech")
        def on_agent_speech(event):
            """Alternative handler for agent speech events"""
            logger.debug(f"agent_speech event received: {type(event).__name__}")
            on_agent_response(event)
        
        start_time = time.time()
        logger.info("Starting session with conversation logging...")
        
        await session.start(
            room=ctx.room,
            agent=agent,
            room_input_options=RoomInputOptions(
                # For telephony applications, use `BVCTelephony` instead for best results
                noise_cancellation=noise_cancellation.BVCTelephony(),
            ),
        )
        
        total_time = time.time() - entrypoint_start
        start_elapsed = time.time() - start_time
        logger.info(f"Session started successfully in {start_elapsed:.3f}s")
        logger.info(f"Total entrypoint time: {total_time:.3f}s")
        logger.info("=" * 60)
        
    except Exception as e:
        elapsed = time.time() - entrypoint_start
        logger.error(f"Error in entrypoint after {elapsed:.3f}s: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))

