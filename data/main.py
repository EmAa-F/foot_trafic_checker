from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
from enum import Enum
import zipfile
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Transport Data Generator Service",
    description="Generate footfall/traffic data for transport infrastructure",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://v0-transport-prediction-service.vercel.app",
        "https://v0-transport-prediction-service.vercel.app/",
        "https://foot-trafic-checker.onrender.com",
        "https://foot-trafic-checker.onrender.com/",
        "http://localhost:3000",  # For local development
        "http://localhost:5173",  # For Vite local development
        "http://localhost:8000",  # For local backend
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Constants
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

BASE_FOOTFALL = {
    "metro": 5000,
    "railway": 8000,
    "bus": 3000,
    "signal": 4000
}

LOCATION_MULTIPLIERS = {
    "Ghatkopar": 1.2, "Andheri": 1.5, "Versova": 0.9, "Aarey": 0.7,
    "Dahisar East": 0.8, "DN Nagar": 1.0, "Azad Nagar": 0.85,
    "Western Express Highway": 1.1, "Marol Naka": 1.3, "Airport Road": 1.4,
    "Andheri West": 1.6, "Bandra": 1.7, "Dadar": 1.8, "Borivali": 1.5,
    "Malad": 1.3, "Goregaon": 1.2, "Jogeshwari": 1.1, "Vile Parle": 1.3,
    "Santa Cruz": 1.2, "Khar Road": 1.4, "Andheri Bus Depot": 1.2,
    "Kurla Bus Station": 1.3, "Borivali Bus Depot": 1.1,
    "Ghatkopar Bus Stand": 1.0, "Bandra Bus Stand": 1.2,
    "Dadar Bus Terminal": 1.4, "Malad Bus Depot": 1.0,
    "Goregaon Bus Depot": 0.9, "Versova Bus Stand": 0.8,
    "DN Nagar Bus Stop": 0.9, "Amboli Naka": 1.3,
    "Andheri Station Signal": 1.5, "Western Express Highway Junction": 1.6,
    "Versova Junction": 1.0, "DN Nagar Signal": 0.9,
    "Azad Nagar Junction": 0.85, "Ghatkopar Junction": 1.2,
    "Airport Road Signal": 1.4, "Marol Naka Signal": 1.3, "Aarey Signal": 0.7
}

# Enums
class TransportType(str, Enum):
    metro = "metro"
    railway = "railway"
    bus = "bus"
    signal = "signal"

# Pydantic models
class DataGenerationRequest(BaseModel):
    location_name: str = Field(..., description="Name of the location")
    transport_type: TransportType = Field(..., description="Type of transport infrastructure")
    days: int = Field(default=90, ge=1, le=365, description="Number of days of historical data")

class BulkGenerationRequest(BaseModel):
    days: int = Field(default=90, ge=1, le=365, description="Number of days of historical data")
    include_metro: bool = Field(default=True)
    include_railway: bool = Field(default=True)
    include_bus: bool = Field(default=True)
    include_signals: bool = Field(default=True)

class LocationInfo(BaseModel):
    name: str
    type: str
    base_footfall: int
    multiplier: float

# Helper functions
def generate_transport_data(location_name: str, transport_type: str, days: int = 90) -> pd.DataFrame:
    """Generate realistic footfall/traffic data for a location"""
    data = []
    start_date = datetime.now() - timedelta(days=days)
    
    base = BASE_FOOTFALL[transport_type]
    multiplier = LOCATION_MULTIPLIERS.get(location_name, 1.0)
    
    for day in range(days):
        current_date = start_date + timedelta(days=day)
        base_footfall = base * multiplier
        
        if current_date.weekday() < 5:
            base_footfall *= 1.3
        else:
            base_footfall *= 0.7
        
        month_factor = 1 + (day / days) * 0.2
        base_footfall *= month_factor
        
        week_pattern = np.sin(2 * np.pi * current_date.weekday() / 7) * (base * 0.1)
        noise = np.random.normal(0, base * 0.05)
        
        if np.random.random() < 0.05:
            event_boost = np.random.uniform(1.2, 1.5)
            base_footfall *= event_boost
        
        footfall = int(base_footfall + week_pattern + noise)
        footfall = max(int(base * 0.2), footfall)
        
        data.append({
            'ds': current_date.strftime('%Y-%m-%d'),
            'y': footfall
        })
    
    return pd.DataFrame(data)

def cleanup_files(filenames: List[str]):
    """Clean up generated files"""
    for filename in filenames:
        try:
            if os.path.exists(filename):
                os.remove(filename)
                logger.info(f"Cleaned up file: {filename}")
        except Exception as e:
            logger.error(f"Error removing file {filename}: {e}")

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Transport Data Generator Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "locations": "/api/locations",
            "generate_data": "/api/generate/data",
            "generate_csv": "/api/generate/csv",
            "generate_all": "/api/generate/all"
        }
    }

@app.get("/api/locations")
async def get_all_locations():
    """Get all available locations with their details"""
    return {
        "metro_stations": [
            {
                "name": station,
                "type": "metro",
                "base_footfall": BASE_FOOTFALL["metro"],
                "multiplier": LOCATION_MULTIPLIERS.get(station, 1.0)
            } for station in METRO_STATIONS
        ],
        "railway_stations": [
            {
                "name": station,
                "type": "railway",
                "base_footfall": BASE_FOOTFALL["railway"],
                "multiplier": LOCATION_MULTIPLIERS.get(station, 1.0)
            } for station in RAILWAY_STATIONS
        ],
        "bus_stations": [
            {
                "name": station,
                "type": "bus",
                "base_footfall": BASE_FOOTFALL["bus"],
                "multiplier": LOCATION_MULTIPLIERS.get(station, 1.0)
            } for station in BUS_STATIONS
        ],
        "traffic_signals": [
            {
                "name": signal,
                "type": "signal",
                "base_footfall": BASE_FOOTFALL["signal"],
                "multiplier": LOCATION_MULTIPLIERS.get(signal, 1.0)
            } for signal in TRAFFIC_SIGNALS
        ]
    }

@app.post("/api/generate/data")
async def generate_data(request: DataGenerationRequest):
    """Generate data for a location and return as JSON"""
    try:
        logger.info(f"Generating data for {request.location_name} ({request.transport_type})")
        
        df = generate_transport_data(
            request.location_name,
            request.transport_type.value,
            request.days
        )
        
        stats = {
            "mean": float(df['y'].mean()),
            "median": float(df['y'].median()),
            "min": int(df['y'].min()),
            "max": int(df['y'].max()),
            "std": float(df['y'].std())
        }
        
        return {
            "success": True,
            "location": request.location_name,
            "transport_type": request.transport_type.value,
            "days": request.days,
            "records": len(df),
            "statistics": stats,
            "data": df.to_dict(orient='records')
        }
    
    except Exception as e:
        logger.error(f"Error generating data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate/csv")
async def generate_csv(
    request: DataGenerationRequest,
    background_tasks: BackgroundTasks
):
    """Generate data for a location and save as CSV file"""
    try:
        logger.info(f"Generating CSV for {request.location_name} ({request.transport_type})")
        
        df = generate_transport_data(
            request.location_name,
            request.transport_type.value,
            request.days
        )
        
        filename = f"data_{request.transport_type.value}_{request.location_name.replace(' ', '_').lower()}.csv"
        df.to_csv(filename, index=False)
        
        logger.info(f"Created CSV file: {filename}")
        
        background_tasks.add_task(cleanup_files, [filename])
        
        return FileResponse(
            path=filename,
            filename=filename,
            media_type="text/csv"
        )
    
    except Exception as e:
        logger.error(f"Error generating CSV: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate/all")
async def generate_all_data(request: BulkGenerationRequest):
    """Generate data for all locations and save as CSV files"""
    try:
        logger.info("Starting bulk data generation")
        generated_files = []
        
        if request.include_metro:
            for station in METRO_STATIONS:
                df = generate_transport_data(station, "metro", request.days)
                filename = f"data_metro_{station.replace(' ', '_').lower()}.csv"
                df.to_csv(filename, index=False)
                generated_files.append(filename)
                logger.info(f"Generated: {filename}")
        
        if request.include_railway:
            for station in RAILWAY_STATIONS:
                df = generate_transport_data(station, "railway", request.days)
                filename = f"data_railway_{station.replace(' ', '_').lower()}.csv"
                df.to_csv(filename, index=False)
                generated_files.append(filename)
                logger.info(f"Generated: {filename}")
        
        if request.include_bus:
            for station in BUS_STATIONS:
                df = generate_transport_data(station, "bus", request.days)
                filename = f"data_bus_{station.replace(' ', '_').lower()}.csv"
                df.to_csv(filename, index=False)
                generated_files.append(filename)
                logger.info(f"Generated: {filename}")
        
        if request.include_signals:
            for signal in TRAFFIC_SIGNALS:
                df = generate_transport_data(signal, "signal", request.days)
                filename = f"data_signal_{signal.replace(' ', '_').lower()}.csv"
                df.to_csv(filename, index=False)
                generated_files.append(filename)
                logger.info(f"Generated: {filename}")
        
        logger.info(f"Bulk generation complete. Total files: {len(generated_files)}")
        
        return {
            "success": True,
            "message": "All data files generated successfully",
            "total_files": len(generated_files),
            "files": generated_files,
            "days_per_file": request.days
        }
    
    except Exception as e:
        logger.error(f"Error in bulk generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate/zip")
async def generate_zip(request: BulkGenerationRequest):
    """Generate all data and return as ZIP file"""
    try:
        logger.info("Generating ZIP archive")
        
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            if request.include_metro:
                for station in METRO_STATIONS:
                    df = generate_transport_data(station, "metro", request.days)
                    filename = f"data_metro_{station.replace(' ', '_').lower()}.csv"
                    csv_data = df.to_csv(index=False)
                    zip_file.writestr(filename, csv_data)
            
            if request.include_railway:
                for station in RAILWAY_STATIONS:
                    df = generate_transport_data(station, "railway", request.days)
                    filename = f"data_railway_{station.replace(' ', '_').lower()}.csv"
                    csv_data = df.to_csv(index=False)
                    zip_file.writestr(filename, csv_data)
            
            if request.include_bus:
                for station in BUS_STATIONS:
                    df = generate_transport_data(station, "bus", request.days)
                    filename = f"data_bus_{station.replace(' ', '_').lower()}.csv"
                    csv_data = df.to_csv(index=False)
                    zip_file.writestr(filename, csv_data)
            
            if request.include_signals:
                for signal in TRAFFIC_SIGNALS:
                    df = generate_transport_data(signal, "signal", request.days)
                    filename = f"data_signal_{signal.replace(' ', '_').lower()}.csv"
                    csv_data = df.to_csv(index=False)
                    zip_file.writestr(filename, csv_data)
        
        zip_buffer.seek(0)
        
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=transport_data.zip"}
        )
    
    except Exception as e:
        logger.error(f"Error creating ZIP: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "data-generator",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Data Generator Service on port 8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
