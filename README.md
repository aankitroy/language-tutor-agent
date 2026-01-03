# Language Tutor Agent

A conversation-based language learning agent built with LiveKit that helps users learn new languages through interactive voice conversations.

## Features

- **Conversation-Based Learning**: Learn languages through natural, interactive conversations
- **Multi-Language Support**: Learn any language (Spanish, French, German, Japanese, etc.)
- **Progress Tracking**: Tracks vocabulary learned, topics covered, and conversation sessions
- **Adaptive Learning**: Adjusts to user's proficiency level (beginner, intermediate, advanced)
- **Real-World Scenarios**: Practice conversations in practical scenarios like restaurants, shopping, directions
- **Gentle Corrections**: Provides feedback and corrections with explanations

## Setup

1. **Create and activate virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -e .
   ```

3. **Configure environment variables**:
   Create a `.env.local` file with:
   ```
   LIVEKIT_URL=wss://your-livekit-server.com
   LIVEKIT_API_KEY=your-api-key
   LIVEKIT_API_SECRET=your-api-secret
   GOOGLE_API_KEY=your-google-api-key
   ```

## Usage

Run the agent:
```bash
python agent.py dev
```

## How It Works

1. **Initial Greeting**: The agent greets the user and asks which language they want to learn
2. **Session Start**: Once a language is selected, a learning session begins
3. **Conversation Practice**: The agent conducts natural conversations in the target language
4. **Vocabulary Introduction**: New words and phrases are introduced gradually
5. **Feedback & Corrections**: The agent provides gentle corrections and explanations
6. **Progress Tracking**: Learning progress is tracked throughout the session

## Agent Tools

- `start_learning_session`: Starts a new learning session for a specific language
- `introduce_vocabulary`: Introduces new vocabulary words by topic
- `practice_conversation`: Starts practice conversations in different scenarios
- `provide_feedback`: Provides corrections and explanations
- `get_progress_summary`: Shows learning progress summary

## Learning Modes

- **Conversation**: Natural back-and-forth conversations
- **Vocabulary**: Focused vocabulary learning
- **Grammar**: Grammar explanations and practice
- **Pronunciation**: Pronunciation practice and feedback

## Example Conversation Flow

1. User: "I want to learn Spanish"
2. Agent: "Great! Let's start learning Spanish. I'll help you learn through conversation."
3. Agent: "Let's begin with greetings. In Spanish, 'Hello' is 'Hola'. Can you say 'Hola'?"
4. User: "Hola"
5. Agent: "Excellent! Now let's practice a simple conversation..."

## Requirements

- Python 3.11+
- LiveKit account and credentials
- Google API key (for Gemini Realtime)

