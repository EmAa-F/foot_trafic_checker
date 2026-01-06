from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import httpx
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

handler = RotatingFileHandler('prediction_api.log', maxBytes=10000000, backupCount=3)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Initialize FastAPI app
app = FastAPI(
    title="Transport Prediction Service",
    description="Congestion prediction and analysis for transport infrastructure",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://v0-transport-prediction-service.vercel.app",
        "https://v0-transport-prediction-service.vercel.app/",
        "http://localhost:3000",  
        "http://localhost:5173",  
    ],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)
# Configuration
DATA_GEN_SERVICE_URL = "https://traffic-data-genrator.onrender.com"  # Data Generator Service URL
REQUEST_TIMEOUT = 30.0

# Transport infrastructure data
METRO_STATIONS = [
    "Ghatkopar", "Andheri", "Versova", "Aarey", "Dahisar East",
    "DN Nagar", "Azad Nagar", "Western Express Highway", "Marol Naka", "Airport Road"
]

RAILWAY_STATIONS = [
    "Andheri West", "Bandra", "Dadar", "Borivali", "Malad", 
    "Goregaon", "Jogeshwari", "Vile Parle", "Santa Cruz", "Khar Road"
]

BUS_STATIONS = [
    "Andheri Bus Depot", "Kurla Bus Station", "Borivali Bus Depot",
    "Ghatkopar Bus Stand", "Bandra Bus Stand", "Dadar Bus Terminal",
    "Malad Bus Depot", "Goregaon Bus Depot", "Versova Bus Stand", "DN Nagar Bus Stop"
]

TRAFFIC_SIGNALS = [
    "Amboli Naka", "Andheri Station Signal", "Western Express Highway Junction",
    "Versova Junction", "DN Nagar Signal", "Azad Nagar Junction",
    "Ghatkopar Junction", "Airport Road Signal", "Marol Naka Signal", "Aarey Signal"
]

# Area-based grouping for congestion analysis
AREA_MAPPING = {
    "Andheri": {
        "metro": ["Andheri"],
        "railway": ["Andheri West"],
        "bus": ["Andheri Bus Depot"],
        "signals": ["Andheri Station Signal", "Amboli Naka"]
    },
    "Versova": {
        "metro": ["Versova"],
        "railway": [],
        "bus": ["Versova Bus Stand"],
        "signals": ["Versova Junction"]
    },
    "Ghatkopar": {
        "metro": ["Ghatkopar"],
        "railway": [],
        "bus": ["Ghatkopar Bus Stand"],
        "signals": ["Ghatkopar Junction"]
    },
    "DN Nagar": {
        "metro": ["DN Nagar"],
        "railway": [],
        "bus": ["DN Nagar Bus Stop"],
        "signals": ["DN Nagar Signal"]
    },
    "Bandra": {
        "metro": [],
        "railway": ["Bandra"],
        "bus": ["Bandra Bus Stand"],
        "signals": []
    },
    "Dadar": {
        "metro": [],
        "railway": ["Dadar"],
        "bus": ["Dadar Bus Terminal"],
        "signals": []
    },
    "Borivali": {
        "metro": [],
        "railway": ["Borivali"],
        "bus": ["Borivali Bus Depot"],
        "signals": []
    }
}

# Hour-based multipliers for rush hours
HOUR_MULTIPLIERS = {
    0: 0.3, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.3, 5: 0.4,
    6: 0.5, 7: 0.8, 8: 1.3, 9: 1.5, 10: 1.2, 11: 1.0,
    12: 0.9, 13: 0.8, 14: 0.7, 15: 0.8, 16: 0.9, 17: 1.2,
    18: 1.5, 19: 1.4, 20: 1.2, 21: 0.9, 22: 0.7, 23: 0.5
}

# Transport type specific multipliers
TRANSPORT_MULTIPLIERS = {
    "metro": 1.0,
    "railway": 1.3,
    "bus": 0.8,
    "signal": 1.2
}

# Pydantic models
class GenerateDataRequest(BaseModel):
    days: int = Field(default=90, ge=1, le=365)
    include_metro: bool = True
    include_railway: bool = True
    include_bus: bool = True
    include_signals: bool = True

# Helper functions
def get_transport_type(location_name: str) -> Optional[str]:
    """Determine transport type for a location"""
    if location_name in METRO_STATIONS:
        return "metro"
    elif location_name in RAILWAY_STATIONS:
        return "railway"
    elif location_name in BUS_STATIONS:
        return "bus"
    elif location_name in TRAFFIC_SIGNALS:
        return "signal"
    return None

async def call_data_generator(endpoint: str, method: str = "GET", json_data: dict = None):
    """Make HTTP call to data generator service"""
    try:
        url = f"{DATA_GEN_SERVICE_URL}{endpoint}"
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            if method == "GET":
                response = await client.get(url)
            elif method == "POST":
                response = await client.post(url, json=json_data)
            
            response.raise_for_status()
            return response.json()
    except httpx.RequestError as e:
        logger.error(f"Request error calling data generator: {e}")
        raise HTTPException(status_code=503, detail="Data generator service unavailable")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from data generator: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))

def calculate_congestion_level(footfall: int, transport_type: str, multiplier: float) -> str:
    """Calculate congestion level based on footfall and transport type"""
    transport_factor = TRANSPORT_MULTIPLIERS.get(transport_type, 1.0)
    adjusted_footfall = footfall * transport_factor
    
    if multiplier > 1.2 and adjusted_footfall > 5000:
        return 'High'
    elif multiplier > 0.9 and adjusted_footfall > 3000:
        return 'Medium'
    else:
        return 'Low'

async def get_location_data(location_name: str, transport_type: str, days: int = 90):
    """Get data for a specific location from data generator service"""
    try:
        data = await call_data_generator(
            "/api/generate/data",
            method="POST",
            json_data={
                "location_name": location_name,
                "transport_type": transport_type,
                "days": days
            }
        )
        return data
    except Exception as e:
        logger.error(f"Error getting data for {location_name}: {e}")
        return None

def calculate_current_footfall(base_footfall: float) -> int:
    """Calculate current footfall based on time of day"""
    current_hour = datetime.now().hour
    multiplier = HOUR_MULTIPLIERS.get(current_hour, 0.8)
    return int(base_footfall * multiplier), multiplier

# API Routes
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Transport Prediction Service",
        "version": "1.0.0",
        "status": "running",
        "data_generator": DATA_GEN_SERVICE_URL,
        "endpoints": {
            "locations": "/api/locations",
            "location_prediction": "/api/location/{location_name}",
            "area_prediction": "/api/area/{area_name}",
            "all_areas": "/api/areas/all",
            "transport_type": "/api/transport/{transport_type}",
            "generate_data": "/api/data/generate"
        }
    }

@app.get("/api/locations")
async def get_all_locations():
    """Get all locations grouped by transport type"""
    return {
        "metro_stations": METRO_STATIONS,
        "railway_stations": RAILWAY_STATIONS,
        "bus_stations": BUS_STATIONS,
        "traffic_signals": TRAFFIC_SIGNALS,
        "areas": list(AREA_MAPPING.keys())
    }

@app.get("/api/location/{location_name}")
async def get_location_prediction(location_name: str):
    """Get detailed prediction for specific location"""
    transport_type = get_transport_type(location_name)
    
    if not transport_type:
        raise HTTPException(status_code=404, detail="Location not found")
    
    try:
        # Get data from data generator service
        data = await get_location_data(location_name, transport_type)
        
        if not data or not data.get("success"):
            raise HTTPException(status_code=404, detail="Data not available")
        
        # Calculate current footfall based on time
        base_footfall = data["statistics"]["mean"]
        current_footfall, multiplier = calculate_current_footfall(base_footfall)
        
        # Calculate congestion
        congestion = calculate_congestion_level(current_footfall, transport_type, multiplier)
        
        return {
            "location": location_name,
            "transport_type": transport_type,
            "current_footfall": current_footfall,
            "base_daily_footfall": int(base_footfall),
            "congestion_level": congestion,
            "statistics": data["statistics"],
            "timestamp": datetime.now().isoformat(),
            "current_hour": datetime.now().hour
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in location prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/area/{area_name}")
async def get_area_prediction(area_name: str):
    """Get comprehensive area congestion analysis"""
    if area_name not in AREA_MAPPING:
        raise HTTPException(status_code=404, detail="Area not found")
    
    try:
        area_data = AREA_MAPPING[area_name]
        current_hour = datetime.now().hour
        multiplier = HOUR_MULTIPLIERS.get(current_hour, 0.8)
        
        congestion_scores = []
        components = []
        
        # Collect data for all locations in the area
        for transport_type, locations in area_data.items():
            for location in locations:
                data = await get_location_data(location, transport_type)
                
                if data and data.get("success"):
                    base_footfall = data["statistics"]["mean"]
                    current_footfall = int(base_footfall * multiplier)
                    congestion = calculate_congestion_level(current_footfall, transport_type, multiplier)
                    
                    score = {'High': 3, 'Medium': 2, 'Low': 1}.get(congestion, 1)
                    congestion_scores.append(score)
                    
                    components.append({
                        'location': location,
                        'type': transport_type,
                        'footfall': current_footfall,
                        'congestion': congestion
                    })
        
        if not congestion_scores:
            raise HTTPException(status_code=404, detail="No data available for this area")
        
        # Calculate overall congestion
        avg_score = sum(congestion_scores) / len(congestion_scores)
        
        if avg_score >= 2.5:
            overall_congestion = 'High'
        elif avg_score >= 1.5:
            overall_congestion = 'Medium'
        else:
            overall_congestion = 'Low'
        
        return {
            'area': area_name,
            'overall_congestion': overall_congestion,
            'congestion_score': round(avg_score, 2),
            'timestamp': datetime.now().isoformat(),
            'current_hour': current_hour,
            'components': components
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in area prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/areas/all")
async def get_all_areas_congestion():
    """Get congestion data for all areas"""
    results = []
    
    for area_name in AREA_MAPPING.keys():
        try:
            area_data = await get_area_prediction(area_name)
            results.append(area_data)
        except Exception as e:
            logger.warning(f"Error getting data for area {area_name}: {e}")
            continue
    
    return {
        "areas": results,
        "total_areas": len(results),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/transport/{transport_type}")
async def get_transport_type_data(transport_type: str):
    """Get all locations data for a specific transport type"""
    type_map = {
        'metro': METRO_STATIONS,
        'railway': RAILWAY_STATIONS,
        'bus': BUS_STATIONS,
        'signal': TRAFFIC_SIGNALS
    }
    
    locations = type_map.get(transport_type)
    if not locations:
        raise HTTPException(status_code=404, detail="Invalid transport type")
    
    try:
        current_hour = datetime.now().hour
        multiplier = HOUR_MULTIPLIERS.get(current_hour, 0.8)
        
        results = []
        
        for location in locations:
            data = await get_location_data(location, transport_type)
            
            if data and data.get("success"):
                base_footfall = data["statistics"]["mean"]
                current_footfall = int(base_footfall * multiplier)
                congestion = calculate_congestion_level(current_footfall, transport_type, multiplier)
                
                results.append({
                    'location': location,
                    'footfall': current_footfall,
                    'congestion': congestion
                })
        
        return {
            'transport_type': transport_type,
            'locations': results,
            'total_locations': len(results),
            'timestamp': datetime.now().isoformat(),
            'current_hour': current_hour
        }
    
    except Exception as e:
        logger.error(f"Error getting transport type data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/data/generate")
async def trigger_data_generation(request: GenerateDataRequest):
    """Trigger data generation in data generator service"""
    try:
        result = await call_data_generator(
            "/api/generate/all",
            method="POST",
            json_data={
                "days": request.days,
                "include_metro": request.include_metro,
                "include_railway": request.include_railway,
                "include_bus": request.include_bus,
                "include_signals": request.include_signals
            }
        )
        
        return {
            "success": True,
            "message": "Data generation completed successfully",
            "result": result
        }
    
    except Exception as e:
        logger.error(f"Error triggering data generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/generator/status")
async def check_data_generator_status():
    """Check if data generator service is running"""
    try:
        response = await call_data_generator("/health")
        return {
            "data_generator_status": "online",
            "data_generator_response": response
        }
    except Exception as e:
        return {
            "data_generator_status": "offline",
            "error": str(e)
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check data generator service
        data_gen_status = await check_data_generator_status()
        
        return {
            "status": "healthy",
            "service": "prediction-service",
            "timestamp": datetime.now().isoformat(),
            "dependencies": {
                "data_generator": data_gen_status["data_generator_status"]
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "degraded",
            "service": "prediction-service",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Prediction Service on port 8000")
    logger.info(f"Data Generator Service URL: {DATA_GEN_SERVICE_URL}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
