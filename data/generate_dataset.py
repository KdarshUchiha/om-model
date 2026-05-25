"""
Om-Code Dataset Generator
Generates synthetic training examples using Gemini API by expanding seed examples.
Usage: python generate_dataset.py YOUR_GEMINI_API_KEY [--count 500] [--output training_data.jsonl]
"""

import argparse
import json
import time
import random
import sys
from pathlib import Path

try:
    import google.generativeai as genai
except ImportError:
    print("Install google-generativeai: pip install google-generativeai")
    sys.exit(1)

SYSTEM_PROMPT = (
    "You are Om — The Divine Architect. A polymath AI that builds anything. "
    "You write complete, working, self-contained code. Your style: concise, no unnecessary comments, "
    "single-file HTML+CSS+JS when possible. You think like an architect, code like a craftsman, "
    "and design like an artist."
)

CATEGORIES = [
    {
        "name": "games",
        "prompts": [
            "Build a {game_type} game with {feature}",
            "Create a {game_type} where {mechanic}",
            "Make a browser {game_type} with {visual_style} aesthetics",
            "I want a {game_type} that has {feature} and {mechanic}",
        ],
        "game_types": [
            "snake", "tetris", "breakout", "platformer", "puzzle", "rpg",
            "tower defense", "space shooter", "memory card", "maze",
            "flappy bird clone", "pong", "asteroids", "minesweeper",
            "match-3", "typing game", "rhythm game", "dungeon crawler",
            "racing", "chess", "checkers", "tic-tac-toe variant",
        ],
        "features": [
            "particle effects", "smooth animations", "power-ups",
            "multiple levels", "a score system", "sound effects",
            "dark theme", "neon colors", "pixel art style",
            "touch controls", "leaderboard", "difficulty scaling",
        ],
        "mechanics": [
            "gravity affects all objects", "time slows when you're close to danger",
            "the world wraps around", "enemies learn from your patterns",
            "you can only move in one direction", "colors determine abilities",
            "the map generates procedurally", "objects have physics",
        ],
        "visual_styles": ["cyberpunk", "retro", "minimalist", "neon", "pixel", "vaporwave"],
    },
    {
        "name": "web_apps",
        "prompts": [
            "Build a {app_type} with {feature}",
            "Create a {app_type} that {behavior}",
            "Make a responsive {app_type} with {visual_style} design",
            "I need a {app_type} with {feature} and {behavior}",
        ],
        "app_types": [
            "todo app", "kanban board", "chat interface", "dashboard",
            "landing page", "portfolio site", "markdown editor",
            "calculator", "timer app", "weather widget", "music player UI",
            "file upload interface", "color palette generator",
            "form builder", "data table", "calendar", "image gallery",
            "notification center", "settings panel", "search interface",
        ],
        "features": [
            "drag and drop", "dark mode toggle", "local storage persistence",
            "keyboard shortcuts", "animations", "responsive grid",
            "real-time filtering", "infinite scroll", "toast notifications",
            "modal dialogs", "tab navigation", "breadcrumbs",
        ],
        "behaviors": [
            "saves state to localStorage", "works offline",
            "has smooth transitions between views", "supports themes",
            "shows loading skeletons", "handles errors gracefully",
            "supports multiple languages", "exports data as JSON",
        ],
        "visual_styles": ["glassmorphism", "neumorphism", "brutalist", "minimal", "material"],
    },
    {
        "name": "architecture",
        "prompts": [
            "How would you architect a {system}?",
            "Plan the structure for a {system} that {requirement}",
            "What's the best approach to build a {system}?",
            "Design a {system} with {constraint}",
        ],
        "systems": [
            "real-time multiplayer game", "social media feed", "e-commerce checkout",
            "notification system", "file sync service", "API gateway",
            "plugin system", "state management library", "component library",
            "build tool", "CLI framework", "database ORM",
        ],
        "requirements": [
            "scales to millions of users", "works offline-first",
            "has real-time updates", "supports plugins",
            "is framework-agnostic", "handles concurrent edits",
        ],
        "constraints": [
            "no external dependencies", "under 1000 lines",
            "type-safe throughout", "zero config needed",
            "backwards compatible API", "sub-100ms response time",
        ],
    },
    {
        "name": "debugging",
        "prompts": [
            "This {component} is {problem}. Here's the code: {code_hint}",
            "Why is my {component} {problem}?",
            "Fix this {component} that {problem}",
        ],
        "components": [
            "CSS grid layout", "flexbox container", "event listener",
            "async function", "animation", "form validation",
            "localStorage handler", "fetch request", "canvas drawing",
            "WebSocket connection", "intersection observer", "drag handler",
        ],
        "problems": [
            "not rendering correctly", "causing a memory leak",
            "firing multiple times", "not responsive on mobile",
            "flickering on scroll", "losing state on navigation",
            "blocking the main thread", "not handling errors",
        ],
    },
    {
        "name": "design",
        "prompts": [
            "Design a {element} with {style_desc}",
            "Create CSS for a {element} that {behavior}",
            "Style a {element} with {aesthetic} vibes",
        ],
        "elements": [
            "button component", "card grid", "navigation bar",
            "hero section", "pricing table", "testimonial slider",
            "footer", "sidebar menu", "modal", "tooltip",
            "progress indicator", "avatar group",
        ],
        "style_descs": [
            "smooth hover transitions", "gradient backgrounds",
            "subtle shadows", "glass effect", "animated borders",
            "responsive breakpoints", "fluid typography",
        ],
        "aesthetics": ["cosmic", "ocean", "forest", "sunset", "midnight", "aurora"],
        "behaviors": [
            "morphs on hover", "pulses to draw attention",
            "slides in from the side", "fades between states",
            "responds to scroll position", "adapts to content length",
        ],
    },
]

GENERATION_PROMPT = """You are generating training data for "Om-Code", a code generation AI.

Given this example of Om's style:

USER: {example_user}
ASSISTANT (first 200 chars): {example_assistant_preview}

Now generate a NEW example. The user asks: "{new_prompt}"

Write Om's complete response. Rules:
1. Write COMPLETE, WORKING code (full HTML file with embedded CSS and JS)
2. No placeholders, no "..." — everything must work if pasted into a browser
3. Keep it 200-500 lines
4. Om's voice: confident, direct, no unnecessary comments
5. Use Sanskrit/Hindu-inspired variable/function names occasionally (like dharma, karma, shakti, maya, prana)
6. Self-contained single HTML file
7. Start directly with the code (```html), no preamble

Respond ONLY with the assistant's message (the complete code). No meta-commentary."""


def load_seeds(path: str) -> list:
    examples = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def generate_prompt(category: dict) -> str:
    template = random.choice(category["prompts"])
    kwargs = {}
    for key in template.split("{")[1:]:
        field = key.split("}")[0]
        if field in category:
            kwargs[field] = random.choice(category[field])
        else:
            kwargs[field] = "something cool"
    return template.format(**kwargs)


def generate_examples(api_key: str, seeds: list, count: int, output_path: str):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    generated = []
    attempts = 0
    max_attempts = count * 3

    print(f"Generating {count} examples...")

    while len(generated) < count and attempts < max_attempts:
        attempts += 1
        category = random.choice(CATEGORIES)
        new_prompt = generate_prompt(category)
        seed = random.choice(seeds)

        seed_user = next(m["content"] for m in seed["messages"] if m["role"] == "user")
        seed_assistant = next(m["content"] for m in seed["messages"] if m["role"] == "assistant")

        prompt = GENERATION_PROMPT.format(
            example_user=seed_user,
            example_assistant_preview=seed_assistant[:200],
            new_prompt=new_prompt,
        )

        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.8,
                    max_output_tokens=8192,
                ),
            )

            assistant_content = response.text.strip()
            if assistant_content.startswith("```html"):
                assistant_content = assistant_content[7:]
            if assistant_content.startswith("```"):
                assistant_content = assistant_content[3:]
            if assistant_content.endswith("```"):
                assistant_content = assistant_content[:-3]
            assistant_content = assistant_content.strip()

            if len(assistant_content) < 500:
                print(f"  Skipping short response ({len(assistant_content)} chars)")
                continue

            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": new_prompt},
                    {"role": "assistant", "content": assistant_content},
                ]
            }
            generated.append(example)
            print(f"  [{len(generated)}/{count}] Generated: {new_prompt[:60]}...")

            time.sleep(1.0)

        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(5.0)
            continue

    with open(output_path, "w") as f:
        for seed in seeds:
            f.write(json.dumps(seed) + "\n")
        for ex in generated:
            f.write(json.dumps(ex) + "\n")

    total = len(seeds) + len(generated)
    print(f"\nDone! Wrote {total} examples ({len(seeds)} seeds + {len(generated)} generated) to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Om-Code training data")
    parser.add_argument("api_key", help="Gemini API key")
    parser.add_argument("--count", type=int, default=500, help="Number of examples to generate")
    parser.add_argument("--seeds", default=str(Path(__file__).parent / "seed_examples.jsonl"), help="Path to seed examples")
    parser.add_argument("--output", default=str(Path(__file__).parent / "training_data.jsonl"), help="Output path")
    args = parser.parse_args()

    seeds = load_seeds(args.seeds)
    print(f"Loaded {len(seeds)} seed examples")

    generate_examples(args.api_key, seeds, args.count, args.output)


if __name__ == "__main__":
    main()
