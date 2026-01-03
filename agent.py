import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, Optional
from dataclasses import dataclass, field

from google.genai.types import HttpOptions

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
                
                return LearningSession(
                    target_language=document.get("target_language", ""),
                    current_level=document.get("current_level", "beginner"),
                    topics_covered=document.get("topics_covered", []),
                    vocabulary_learned=document.get("vocabulary_learned", []),
                    conversation_count=document.get("conversation_count", 0),
                    practice_mode=document.get("practice_mode", "conversation"),
                    user_id=user_id
                )
            else:
                logger.info(f"🆕 New user detected: {user_id}")
                return None
        except Exception as e:
            logger.error(f"Error loading session: {e}", exc_info=True)
            return None
    
    def save_session(self, user_id: str, session: 'LearningSession'):
        """Save LearningSession to ChromaDB"""
        try:
            # Prepare document (full session data as source of truth)
            document = {
                "target_language": session.target_language,
                "current_level": session.current_level,
                "topics_covered": session.topics_covered,
                "vocabulary_learned": session.vocabulary_learned,
                "conversation_count": session.conversation_count,
                "practice_mode": session.practice_mode,
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
            
            logger.info(f"💾 Saved session for user {user_id}: {session.target_language} - {session.current_level}")
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


class LanguageTutorAgent(Agent):
    def __init__(self) -> None:
        init_start = time.time()
        logger.info("Initializing LanguageTutorAgent...")
        
        super().__init__(
            instructions="""You are a friendly and patient language learning tutor. Your goal is to help users learn a new language through conversation-based learning.

CRITICAL PRONUNCIATION RULE: When teaching words or phrases in the target language (especially Telugu, Hindi, or any non-English language), you MUST pronounce them using the NATIVE pronunciation and accent of that language. Do NOT use English pronunciation for non-English words.

For example:
- If teaching Telugu, pronounce Telugu words like "నమస్కారం" (Namaskaram), "ఎలా ఉన్నారు" (Ela unnaru), "ధన్యవాదాలు" (Dhanyavadalu) with proper Telugu pronunciation and accent, NOT English accent
- If teaching Spanish, pronounce "Hola" with Spanish pronunciation, not English
- If teaching French, pronounce "Bonjour" with French pronunciation, not English
- Always use the native accent and pronunciation for the target language words

When a user first connects:
1. Greet them warmly and ask which language they would like to learn
2. Once they specify a language, confirm it and start the learning session

During the learning session:
- Conduct natural conversations in the target language
- Start with simple greetings and basic phrases
- When speaking words/phrases in the target language, use NATIVE pronunciation and accent - this is CRITICAL
- When explaining in English, use English pronunciation
- DO NOT repeat phrases with English translations in parentheses - speak naturally in the target language
- If you need to explain meaning, do it separately in English, not inline with the target language phrase
- For example, say "నమస్కారం" naturally, then separately explain "That means 'Hello' in Telugu" - don't say "నమస్కారం (Namaskaram, Hello)"
- Gradually introduce new vocabulary and phrases with proper native pronunciation
- Correct mistakes gently and provide explanations
- Encourage the user to practice speaking with native pronunciation
- Use English to explain concepts when needed, but prioritize using the target language with native pronunciation
- Make learning fun and engaging with real-world scenarios (ordering food, asking directions, etc.)
- Track their progress and adjust difficulty accordingly

Be patient, encouraging, and adapt to the user's learning pace. Celebrate their progress and make them feel comfortable making mistakes.

REMEMBER: Always use native pronunciation and accent for the target language words. Never use English accent for non-English words. This is especially important for Telugu, Hindi, and other Indian languages.

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
    ) -> dict[str, Any]:
        """Start a new language learning session for the specified language.
        
        Args:
            language: The language the user wants to learn (e.g., 'Spanish', 'French', 'German', 'Japanese', 'Telugu', 'Hindi')
            level: The user's proficiency level (beginner, intermediate, advanced)
        """
        start_time = time.time()
        logger.info(f"start_learning_session called - language: {language}, level: {level}")
        
        # Access LearningSession from userdata (source of truth)
        session_data = context.userdata
        
        # Normalize language name
        language = language.strip().title()
        
        # Validate level
        if level not in ["beginner", "intermediate", "advanced"]:
            logger.warning(f"Invalid level '{level}', defaulting to 'beginner'")
            level = "beginner"
        
        # Update the session
        session_data.target_language = language
        session_data.current_level = level
        
        # Save immediately (user_id should always be set from entrypoint)
        if session_data.user_id:
            session_storage.save_session(session_data.user_id, session_data)
        else:
            logger.warning("start_learning_session: user_id not set, progress not saved")
        
        elapsed = time.time() - start_time
        logger.info(f"🌍 LANGUAGE LEARNING SESSION STARTED! Language: {language}, Level: {level} (took {elapsed:.3f}s)")
        
        return {
            "status": "started",
            "language": language,
            "level": level,
            "message": f"Great! Let's start learning {language}. I'll help you learn through conversation using native {language} pronunciation. Let's begin with some basic greetings!",
        }

    @function_tool()
    async def introduce_vocabulary(
        self,
        context: RunContext[LearningSession],
        words: list[str],
        topic: str = "general",
    ) -> dict[str, Any]:
        """Introduce new vocabulary words to the user.
        
        Args:
            words: List of new words/phrases to teach
            topic: The topic category (e.g., 'greetings', 'food', 'directions', 'numbers')
        """
        start_time = time.time()
        logger.debug(f"introduce_vocabulary called - topic: {topic}, words count: {len(words)}")
        
        session_data = context.userdata
        
        if not session_data.target_language:
            logger.warning("introduce_vocabulary called without active session")
            return {
                "status": "error",
                "message": "No learning session active. Please start a session first.",
            }
        
        # Add to vocabulary learned
        session_data.vocabulary_learned.extend(words)
        if topic not in session_data.topics_covered:
            session_data.topics_covered.append(topic)
        
        # Save progress
        if session_data.user_id:
            session_storage.save_session(session_data.user_id, session_data)
        
        elapsed = time.time() - start_time
        logger.info(f"📚 NEW VOCABULARY INTRODUCED! Topic: {topic}, Words: {len(words)} (took {elapsed:.3f}s)")
        logger.debug(f"Words: {', '.join(words)}")
        
        return {
            "status": "success",
            "topic": topic,
            "words": words,
            "message": f"I've introduced {len(words)} new {topic} words. Let's practice using them in conversation!",
        }

    @function_tool()
    async def practice_conversation(
        self,
        context: RunContext[LearningSession],
        scenario: str = "general",
    ) -> dict[str, Any]:
        """Start a practice conversation scenario.
        
        Args:
            scenario: The conversation scenario (e.g., 'restaurant', 'shopping', 'directions', 'introductions')
        """
        start_time = time.time()
        logger.debug(f"practice_conversation called - scenario: {scenario}")
        
        session_data = context.userdata
        
        if not session_data.target_language:
            logger.warning("practice_conversation called without active session")
            return {
                "status": "error",
                "message": "No learning session active. Please start a session first.",
            }
        
        session_data.conversation_count += 1
        if scenario not in session_data.topics_covered:
            session_data.topics_covered.append(scenario)
        
        # Save progress
        if session_data.user_id:
            session_storage.save_session(session_data.user_id, session_data)
        
        scenarios = {
            "restaurant": "Let's practice ordering food at a restaurant!",
            "shopping": "Let's practice shopping and asking about prices!",
            "directions": "Let's practice asking for and giving directions!",
            "introductions": "Let's practice introducing yourself and meeting new people!",
            "general": "Let's have a general conversation to practice!",
        }
        
        scenario_message = scenarios.get(scenario, scenarios["general"])
        
        elapsed = time.time() - start_time
        logger.info(f"💬 PRACTICE CONVERSATION STARTED! Scenario: {scenario}, Session: {session_data.conversation_count} (took {elapsed:.3f}s)")
        
        return {
            "status": "started",
            "scenario": scenario,
            "message": scenario_message,
        }

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
        logger.info(f"📊 LEARNING PROGRESS SUMMARY! Language: {session.target_language}, Level: {session.current_level}, Topics: {len(session.topics_covered)}, Vocabulary: {len(session.vocabulary_learned)}, Conversations: {session.conversation_count} (took {elapsed:.3f}s)")
        
        return {
            "status": "success",
            "language": session.target_language,
            "level": session.current_level,
            "topics_covered": session.topics_covered,
            "vocabulary_count": len(session.vocabulary_learned),
            "conversation_count": session.conversation_count,
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
                greeting_instruction = f"""Welcome the user back warmly. 
                Tell them you see they've been learning {session_data.target_language}. """
                if session_data.topics_covered:
                    greeting_instruction += f"Mention they've covered {len(session_data.topics_covered)} topics so far. "
                if session_data.vocabulary_learned:
                    greeting_instruction += f"Tell them they've learned {len(session_data.vocabulary_learned)} words. "
                greeting_instruction += "Ask if they would like to continue where they left off, or start something new."
                
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
            "min_speech_duration": 0.15,      # Minimum speech duration (seconds) - lower for sensitivity
            "min_silence_duration": 0.5,       # Minimum silence duration (seconds) - higher for slow speakers
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
            language="en-IN",
            vertexai=True,
        )
        
        # Create session with LearningSession as userdata (source of truth)
        session = AgentSession[LearningSession](
            userdata=learning_session,  # Pass loaded/created session as userdata
            video_sampler=VoiceActivityVideoSampler(speaking_fps=0.3, silent_fps=0.2),
            user_away_timeout=5,
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
            """Log user input transcriptions - only FINAL transcriptions"""
            try:
                # Try multiple ways to access transcript
                transcript_text = None
                if hasattr(event, 'transcript'):
                    transcript_text = event.transcript
                elif hasattr(event, 'text'):
                    transcript_text = event.text
                elif hasattr(event, 'message'):
                    transcript_text = str(event.message)
                
                if transcript_text:
                    transcript_text = transcript_text.strip()
                    if transcript_text:
                        is_final = getattr(event, 'is_final', False)
                        
                        # Only log FINAL transcriptions - skip INTERIM completely
                        if is_final:
                            logger.info(f"👤 USER: {transcript_text}")
                        # INTERIM transcriptions are completely skipped (not logged at any level)
            except Exception as e:
                logger.error(f"Error logging user transcript: {e}", exc_info=True)
        
        @session.on("agent_response")
        def on_agent_response(event):
            """Log agent responses"""
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
            except Exception as e:
                logger.error(f"Error logging agent response: {e}", exc_info=True)
        
        # Log conversation items
        @session.on("conversation_item_added")
        def on_conversation_item(event):
            """Log conversation items"""
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
                        else:
                            content = sanitize_text(str(content))
                        
                        if content:
                            logger.info(f"💬 CONVERSATION [{role.upper()}]: {content}")
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

