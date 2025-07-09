from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

# Create FastAPI app instance
app = FastAPI(
    title="My API",
    description="A basic FastAPI skeleton with POST requests",
    version="1.0.0"
)

# Add CORS middleware (for frontend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class UserCreate(BaseModel):
    name: str
    email: str
    age: Optional[int] = None

class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    age: Optional[int] = None
    message: str

class ItemCreate(BaseModel):
    title: str
    description: str
    price: float
    category: str

class ItemResponse(BaseModel):
    id: int
    title: str
    description: str
    price: float
    category: str
    created: bool

# In-memory storage (replace with database in production)
users_db = []
items_db = []
user_counter = 1
item_counter = 1

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to FastAPI!", "status": "running"}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# POST request examples

# 1. Simple POST request
@app.post("/users/", response_model=UserResponse)
async def create_user(user: UserCreate):
    global user_counter
    
    # Check if user already exists
    for existing_user in users_db:
        if existing_user["email"] == user.email:
            raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
    new_user = {
        "id": user_counter,
        "name": user.name,
        "email": user.email,
        "age": user.age
    }
    
    users_db.append(new_user)
    user_counter += 1
    
    return UserResponse(
        id=new_user["id"],
        name=new_user["name"],
        email=new_user["email"],
        age=new_user["age"],
        message="User created successfully"
    )

# 2. POST with path parameter
@app.post("/users/{user_id}/items/", response_model=ItemResponse)
async def create_item_for_user(user_id: int, item: ItemCreate):
    global item_counter
    
    # Check if user exists
    user_exists = any(user["id"] == user_id for user in users_db)
    if not user_exists:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Create new item
    new_item = {
        "id": item_counter,
        "user_id": user_id,
        "title": item.title,
        "description": item.description,
        "price": item.price,
        "category": item.category
    }
    
    items_db.append(new_item)
    item_counter += 1
    
    return ItemResponse(
        id=new_item["id"],
        title=new_item["title"],
        description=new_item["description"],
        price=new_item["price"],
        category=new_item["category"],
        created=True
    )

# 3. POST with query parameters
@app.post("/process-data/")
async def process_data(
    data: dict,
    operation: str = "default",
    include_metadata: bool = False
):
    """
    Process data with optional query parameters
    """
    try:
        result = {
            "processed_data": data,
            "operation": operation,
            "status": "success"
        }
        
        if include_metadata:
            result["metadata"] = {
                "processed_at": "2024-01-01T00:00:00Z",
                "data_size": len(str(data)),
                "operation_type": operation
            }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# 4. POST with file upload simulation
class FileUpload(BaseModel):
    filename: str
    content: str
    file_type: str

@app.post("/upload/")
async def upload_file(file_data: FileUpload):
    """
    Simulate file upload processing
    """
    # Validate file type
    allowed_types = ["csv", "txt", "json"]
    if file_data.file_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"File type '{file_data.file_type}' not allowed. Allowed types: {allowed_types}"
        )
    
    # Process file (simulation)
    processed_result = {
        "filename": file_data.filename,
        "size": len(file_data.content),
        "type": file_data.file_type,
        "status": "uploaded",
        "message": "File processed successfully"
    }
    
    return processed_result

# 5. POST with complex validation
class ComplexRequest(BaseModel):
    user_id: int
    items: List[str]
    options: dict
    priority: str = "normal"

@app.post("/complex-operation/")
async def complex_operation(request: ComplexRequest):
    """
    Handle complex POST request with validation
    """
    # Validate priority
    valid_priorities = ["low", "normal", "high", "urgent"]
    if request.priority not in valid_priorities:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid priority. Must be one of: {valid_priorities}"
        )
    
    # Validate items
    if not request.items:
        raise HTTPException(status_code=400, detail="Items list cannot be empty")
    
    # Process the request
    result = {
        "user_id": request.user_id,
        "processed_items": len(request.items),
        "priority": request.priority,
        "options_applied": list(request.options.keys()),
        "status": "completed"
    }
    
    return result

# GET endpoints for testing
@app.get("/users/")
async def get_users():
    return {"users": users_db}

@app.get("/items/")
async def get_items():
    return {"items": items_db}

# Error handling example
@app.post("/error-demo/")
async def error_demo(data: dict):
    """
    Demonstrate error handling
    """
    if "error" in data:
        raise HTTPException(
            status_code=400,
            detail="Error key found in data"
        )
    
    if "exception" in data:
        raise Exception("This is a test exception")
    
    return {"message": "No errors found", "data": data}

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)