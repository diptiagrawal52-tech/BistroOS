import os
import json
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import google.generativeai as genai
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BistroOS-Backend")

# Load environment variables
load_dotenv()

app = FastAPI(
    title="BistroOS: Virtual Operations Director API",
    description="FastAPI microservice for AI-driven restaurant shift optimization analysis",
    version="1.0.0"
)

# Enable CORS for decoupled frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Data Models (Request) ---

class ShortedItem(BaseModel):
    item_name: str = Field(..., description="Name of the raw material shorted")
    supplier: str = Field(..., description="Name of the supplier")
    quantity_ordered: str = Field(..., description="Quantity ordered")
    quantity_received: str = Field(..., description="Quantity received")
    impact_severity: str = Field(..., description="High, Medium, or Low impact on menu")

class QualityIssue(BaseModel):
    item_name: str = Field(..., description="Name of the item with quality issues")
    issue: str = Field(..., description="Description of the quality issue (e.g. bruised, warm)")
    supplier: str = Field(..., description="Name of the supplier")
    action_taken: str = Field(..., description="Resolution action (e.g. credit requested, rejected)")

class LogisticsData(BaseModel):
    deliveries_received: int = Field(..., description="Total deliveries received during shift")
    shorted_items: List[ShortedItem] = Field(default=[], description="List of items ordered but not delivered")
    quality_issues: List[QualityIssue] = Field(default=[], description="List of items with quality discrepancies")

class LaborData(BaseModel):
    floor_staff_count: int = Field(..., description="Number of floor staff active")
    kitchen_staff_count: int = Field(..., description="Number of kitchen staff active")
    ratio_warning_triggered: bool = Field(default=False, description="Flag if floor-to-kitchen ratio was outside standard bounds")
    bottlenecks: List[str] = Field(default=[], description="List of labor bottlenecks observed (e.g. plate shortage, runners slow)")

class ServiceFlag(BaseModel):
    table_number: str = Field(..., description="Table identifier")
    issue: str = Field(..., description="Description of the service issue")
    resolution: str = Field(..., description="Action taken to recover guest satisfaction")

class CustomerServiceData(BaseModel):
    average_table_turn_mins: int = Field(..., description="Average minutes a table occupied")
    guest_satisfaction_score: float = Field(..., description="Average guest satisfaction score (e.g. 1.0 to 5.0)")
    service_flags: List[ServiceFlag] = Field(default=[], description="List of service standard flags/failures")

class CookTimeline(BaseModel):
    menu_item: str = Field(..., description="Name of the menu item")
    cook_time_mins: int = Field(..., description="Active cook time in minutes")
    prep_time_mins: int = Field(..., description="Prep assembly time in minutes")

class KitchenPrepData(BaseModel):
    prep_completion_percentage: int = Field(..., description="Percentage of prep checklist completed before shift")
    prep_list_gaps: List[str] = Field(default=[], description="Uncompleted items on the prep checklist")
    cooking_timelines: List[CookTimeline] = Field(default=[], description="Active menu items cook times for timeline optimization")

class HighMarginItem(BaseModel):
    item_name: str = Field(..., description="Name of high-margin item")
    quantity: int = Field(..., description="Quantity sold during shift")
    unit_margin: float = Field(..., description="Gross profit margin per unit")

class SalesData(BaseModel):
    revenue: float = Field(..., description="Total gross revenue for the shift")
    covers: int = Field(..., description="Total number of guests served")
    avg_spend_per_head: float = Field(..., description="Calculated average spend per guest")
    high_margin_items_sold: List[HighMarginItem] = Field(default=[], description="List of key high-margin items tracked")

class ShiftDataInput(BaseModel):
    date: str = Field(..., description="Date of the operational shift (YYYY-MM-DD)")
    shift_type: str = Field(..., description="Shift window: Lunch, Dinner, or All-Day")
    logistics: LogisticsData
    labor: LaborData
    customer_service: CustomerServiceData
    kitchen_prep: KitchenPrepData
    sales: SalesData

# --- Pydantic Data Models (Response) ---

class KPISummary(BaseModel):
    logistics_health: str
    labor_efficiency: str
    guest_experience: str
    kitchen_efficiency: str
    financial_health: str

class ShiftAnalysisOutput(BaseModel):
    operational_score: int = Field(..., description="Overall score for the shift, from 0 to 100")
    summary: str = Field(..., description="High-level operational overview from the Virtual Operations Director")
    kpis: KPISummary = Field(..., description="Status breakdown of the 5 core modules")
    critical_red_flags: List[str] = Field(..., description="Immediate problems requiring management intervention")
    actionable_recommendations: List[str] = Field(..., description="Strategic suggestions for the next shift")
    demo_mode: Optional[bool] = Field(default=False, description="Flag highlighting if response was generated via mock mode due to missing API key")

# --- Helper: Smart Mock Generator (Fallback) ---
def generate_smart_mock(data: ShiftDataInput) -> ShiftAnalysisOutput:
    # 1. Base Score calculation
    score = 85
    
    # Reductions
    score -= len(data.logistics.shorted_items) * 4
    score -= len(data.logistics.quality_issues) * 5
    score -= len(data.labor.bottlenecks) * 5
    score -= len(data.customer_service.service_flags) * 6
    
    if data.customer_service.guest_satisfaction_score < 4.0:
        score -= int((4.0 - data.customer_service.guest_satisfaction_score) * 20)
    
    if data.kitchen_prep.prep_completion_percentage < 90:
        score -= int((90 - data.kitchen_prep.prep_completion_percentage) / 2)
        
    if data.labor.ratio_warning_triggered:
        score -= 8
        
    score = max(10, min(100, score))
    
    # 2. Dynamic Red Flags
    red_flags = []
    for item in data.logistics.shorted_items:
        if item.impact_severity == "High":
            red_flags.append(f"CRITICAL SHORTAGE: Ordered {item.quantity_ordered} of {item.item_name} from {item.supplier} but only received {item.quantity_received}.")
    for q in data.logistics.quality_issues:
        red_flags.append(f"QUALITY ALERT: {q.item_name} from {q.supplier} rejected due to '{q.issue}'. Action taken: {q.action_taken}.")
    for b in data.labor.bottlenecks:
        red_flags.append(f"LABOR BOTTLENECK: {b}")
    for flag in data.customer_service.service_flags:
        red_flags.append(f"SERVICE BREAKDOWN (Table {flag.table_number}): {flag.issue}. Resolved by: {flag.resolution}.")
    for gap in data.kitchen_prep.prep_list_gaps:
        red_flags.append(f"PREP GAP: Unfinished prep item: '{gap}'.")
        
    if not red_flags:
        red_flags.append("No critical operational failures reported for this shift.")
        
    # 3. Dynamic Recommendations
    recommendations = []
    if data.logistics.shorted_items or data.logistics.quality_issues:
        suppliers = set([x.supplier for x in data.logistics.shorted_items] + [y.supplier for y in data.logistics.quality_issues])
        recommendations.append(f"Conduct vendor performance review with {', '.join(suppliers)} regarding missing/substandard ingredients.")
        
    if data.labor.floor_staff_count > 0 and data.labor.kitchen_staff_count > 0:
        ratio = data.labor.floor_staff_count / data.labor.kitchen_staff_count
        if ratio > 2.0:
            recommendations.append(f"Floor-to-Kitchen staff ratio ({ratio:.2f}) is high. Floor is overstaffed relative to kitchen speed; shift 1 floor runner to support back-of-house (BOH) prep/expo.")
        elif ratio < 1.0:
            recommendations.append(f"Floor-to-Kitchen staff ratio ({ratio:.2f}) is dangerously low. Kitchen is cooking faster than floor service can run plates. Add at least 1 food runner for the next {data.shift_type} shift.")
            
    if data.customer_service.guest_satisfaction_score < 4.2:
        recommendations.append("Establish a mid-shift hospitality check: Floor manager must table-touch every party within 15 minutes of mains serving.")
        
    if data.kitchen_prep.prep_completion_percentage < 95:
        recommendations.append(f"Shift prep lists to start 30 minutes earlier. Current completion is at {data.kitchen_prep.prep_completion_percentage}%, causing line delays.")
        
    total_margin = sum([item.quantity * item.unit_margin for item in data.sales.high_margin_items_sold])
    if total_margin < 200:
        recommendations.append("Upsell campaign recommended: Coach service staff on suggestive selling for high-margin specials and signature cocktails.")
    else:
        recommendations.append(f"Excellent performance on high-margin items, generating ${total_margin:.2f} in gross profit. Maintain incentive contests for top upsellers.")
        
    if len(recommendations) < 3:
        recommendations.append("Optimize table turnaround times by prioritizing bus-sergeants to clear finished tables within 3 minutes of guest exit.")
        
    # 4. KPI Status
    logistics_status = "Excellent" if not data.logistics.shorted_items else ("Satisfactory" if len(data.logistics.shorted_items) < 3 else "Critical")
    labor_status = "Efficient" if not data.labor.ratio_warning_triggered else "Imbalanced"
    service_status = "Excellent" if data.customer_service.guest_satisfaction_score >= 4.5 else ("Good" if data.customer_service.guest_satisfaction_score >= 4.0 else "Needs Attention")
    kitchen_status = f"High ({data.kitchen_prep.prep_completion_percentage}%)" if data.kitchen_prep.prep_completion_percentage >= 90 else f"Delayed ({data.kitchen_prep.prep_completion_percentage}%)"
    
    avg_spend = data.sales.avg_spend_per_head
    financial_status = f"Strong (${avg_spend:.2f} PPH)" if avg_spend >= 35 else f"Underperforming (${avg_spend:.2f} PPH)"

    return ShiftAnalysisOutput(
        operational_score=score,
        summary=f"Virtual Director Analysis (DEMO MODE): The {data.shift_type} shift on {data.date} generated ${data.sales.revenue:,.2f} in revenue from {data.covers} covers, averaging ${data.sales.avg_spend_per_head:.2f} spend per head. Operational flow scored {score}/100, driven by {kitchen_status.lower()} kitchen prep readiness.",
        kpis=KPISummary(
            logistics_health=logistics_status,
            labor_efficiency=labor_status,
            guest_experience=service_status,
            kitchen_efficiency=kitchen_status,
            financial_health=financial_status
        ),
        critical_red_flags=red_flags,
        actionable_recommendations=recommendations[:4],
        demo_mode=True
    )

# --- Endpoint: Analyze Shift (POST) ---

@app.post("/api/analyze-shift", response_model=ShiftAnalysisOutput)
async def analyze_shift(payload: ShiftDataInput):
    logger.info(f"Received shift data for analysis: {payload.date} ({payload.shift_type})")
    
    api_key = os.getenv("GEMINI_API_KEY")
    # Check if a valid API key was configured
    if not api_key or api_key.strip() == "" or api_key == "your_gemini_api_key_here":
        logger.warning("GEMINI_API_KEY not set. Falling back to Smart Operations Director Mock Generator.")
        return generate_smart_mock(payload)
        
    try:
        # Initialize Google Generative AI
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        # Build prompt
        system_instruction = (
            "You are a highly experienced Principal Restaurant Operations Director, Enterprise Architect, "
            "and Hospitality SaaS Product Consultant with over 15 years of Michelin-starred operational leadership. "
            "Your job is to analyze shift data across 5 operational modules (Logistics, Labor, Service, Prep, Sales) "
            "and return a highly structured shift summary evaluation. "
            "You must return ONLY a JSON response matching the following schema structure exactly:\n"
            "{\n"
            "  \"operational_score\": number (0 to 100 representing operational effectiveness),\n"
            "  \"summary\": \"Concise paragraph outlining overall shift flow, successes, and core concerns.\",\n"
            "  \"kpis\": {\n"
            "    \"logistics_health\": \"Brief status phrase (e.g. Excellent / Satisfactory / Critical with rationale)\",\n"
            "    \"labor_efficiency\": \"Brief status phrase (e.g. Balanced / Imbalanced BOH vs FOH with ratio rationale)\",\n"
            "    \"guest_experience\": \"Brief status phrase (e.g. Excellent / Needs Attention with table turns or complaints mentioned)\",\n"
            "    \"kitchen_efficiency\": \"Brief status phrase (e.g. Excellent prep flow / Delayed prep list)\",\n"
            "    \"financial_health\": \"Brief status phrase (e.g. High margin velocity / Low upsell performance)\"\n"
            "  },\n"
            "  \"critical_red_flags\": [\n"
            "    \"Immediate item requiring manager intervention 1\",\n"
            "    \"Immediate item requiring manager intervention 2\"\n"
            "  ],\n"
            "  \"actionable_recommendations\": [\n"
            "    \"Specific, tactical change for tomorrow's shift 1\",\n"
            "    \"Specific, tactical change for tomorrow's shift 2\",\n"
            "    \"Specific, tactical change for tomorrow's shift 3\"\n"
            "  ]\n"
            "}"
        )
        
        # Format payload to JSON string
        payload_json = json.dumps(payload.model_dump(), indent=2)
        
        prompt = (
            f"Here is the raw operational shift log for the shift on {payload.date} ({payload.shift_type}):\n\n"
            f"```json\n{payload_json}\n```\n\n"
            "Analyze this shift log. Check staff ratios, shorted items impact, service delays, prep list completion, "
            "and financial spending velocities. Give a realistic, high-value, professional assessment. "
            "Provide output in strict JSON format. Do not wrap it in markdown block quotes other than valid json."
        )
        
        response = model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.2
            },
            contents=prompt
        )
        
        response_text = response.text.strip()
        # Clean potential markdown wrapping if returned by Gemini
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "", 1)
        if response_text.endswith("```"):
            response_text = response_text.rsplit("```", 1)[0]
        response_text = response_text.strip()
        
        analysis_data = json.loads(response_text)
        
        # Ensure schema completeness
        output = ShiftAnalysisOutput(
            operational_score=int(analysis_data.get("operational_score", 75)),
            summary=analysis_data.get("summary", "Analysis completed successfully."),
            kpis=KPISummary(
                logistics_health=analysis_data.get("kpis", {}).get("logistics_health", "Normal"),
                labor_efficiency=analysis_data.get("kpis", {}).get("labor_efficiency", "Normal"),
                guest_experience=analysis_data.get("kpis", {}).get("guest_experience", "Normal"),
                kitchen_efficiency=analysis_data.get("kpis", {}).get("kitchen_efficiency", "Normal"),
                financial_health=analysis_data.get("kpis", {}).get("financial_health", "Normal")
            ),
            critical_red_flags=analysis_data.get("critical_red_flags", []),
            actionable_recommendations=analysis_data.get("actionable_recommendations", []),
            demo_mode=False
        )
        return output
        
    except Exception as e:
        logger.error(f"Gemini API analysis failed: {str(e)}")
        # If API configuration is correct but a call fails (e.g. rate limit, auth error, API format change),
        # return the smart mock but add details to the summary so the user knows.
        mocked_res = generate_smart_mock(payload)
        mocked_res.summary = f"Notice: AI Analysis attempted but failed due to error: {str(e)}. Displaying Smart Mocked Director Summary instead."
        return mocked_res

# --- Endpoint: Root Page (GET) ---
# Serves the frontend single-page dashboard statically
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse(
            content="<h1>BistroOS Frontend index.html not found!</h1><p>Please create index.html in the same directory.</p>",
            status_code=404
        )
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read index.html: {str(e)}")

# --- Endpoint: Health Check (GET) ---
@app.get("/api/health")
async def health_check():
    api_key = os.getenv("GEMINI_API_KEY")
    api_configured = bool(api_key and api_key.strip() != "" and api_key != "your_gemini_api_key_here")
    return {
        "status": "healthy",
        "gemini_api_key_configured": api_configured,
        "mode": "Live AI Director" if api_configured else "Local Simulation Engine (Demo)"
    }
