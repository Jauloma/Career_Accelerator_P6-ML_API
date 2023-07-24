from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List

# Create an instance of FastAPI
app = FastAPI()

# In-memory task storage (you might use a database in a real application)
tasks = []

# Model for Task
class Task(BaseModel):
    title: str
    description: str

# Define an endpoint to create a new task
@app.post("/tasks/", response_model=Task)
async def create_task(task: Task):
    tasks.append(task)
    return task

# Define an endpoint to get all tasks
@app.get("/tasks/", response_model=List[Task])
async def get_tasks():
    return tasks

# Define an endpoint to get a specific task by its index in the tasks list
@app.get("/tasks/{task_index}", response_model=Task)
async def get_task(task_index: int = Query(..., ge=0, lt=len(tasks))):
    if task_index >= len(tasks):
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_index]

# Define an endpoint to delete a task by its index
@app.delete("/tasks/{task_index}", response_model=Task)
async def delete_task(task_index: int = Query(..., ge=0, lt=len(tasks))):
    if task_index >= len(tasks):
        raise HTTPException(status_code=404, detail="Task not found")
    deleted_task = tasks.pop(task_index)
    return deleted_task
