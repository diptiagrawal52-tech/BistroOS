# BistroOS: The Integrated Restaurant Operations Hub

BistroOS is a comprehensive, full-stack restaurant operations control center designed for independent restaurant managers to perform shift diagnostics.

## Features
- **Inwarding Logistics**: Track deliveries, monitor shorted items, and manage supplier quality exceptions.
- **Labor & Staffing**: Optimize Front-of-House (FOH) to Back-of-House (BOH) staffing ratios and log shift bottlenecks.
- **Guest Service Experience**: Monitor table turn times, guest satisfaction scores, and log service standard flags with manager recovery actions.
- **Kitchen Prep & Cooking Timelines**: Sequence kitchen prep checklist gaps and automatically sort cooking ticket drop sequences.
- **Sales & Margin Integration**: Track gross shift revenues, average spend per guest head, and high-margin product gross margins.
- **AI-Powered Diagnostics**: Leverage Google Gemini 2.5 Flash to automatically analyze shift logs, generate performance scores, identify red flags, and provide actionable recommendations.

## Tech Stack
- **Backend**: FastAPI (Python)
- **Frontend**: HTML5, Tailwind CSS, JavaScript (Vanilla UI)
- **AI Engine**: Google GenAI (Gemini 2.5 Flash)
- **Deployment**: Docker-ready for Render.com

## Running Locally
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your `GEMINI_API_KEY` in a `.env` file (see `.env.example`).
4. Run the FastAPI application:
   ```bash
   uvicorn main:app --reload
   ```
