# Multi-Agent System

A distributed task processing system built with Python that coordinates multiple specialized agents to handle complex workflows including research, analysis, and question-answering.

## Features

- Multi-agent architecture with specialized roles
- Asynchronous message passing system
- Thread-safe shared memory
- Dynamic task planning and decomposition
- Concurrent agent execution
- Real-time task coordination

## Tech Stack

**Language:** Python 3.7+

**Core Libraries:**
- `threading` (concurrent execution)
- `queue` (message passing)
- `dataclasses` (structured data)
- `json` (data serialization)
- `time` (timing operations)

## Architecture

The system consists of five specialized agents:

| Agent | Responsibility |
|-------|----------------|
| `PlannerAgent` | Task decomposition and workflow planning |
| `ResearcherAgent` | Information gathering and research |
| `SummarizerAgent` | Content summarization and synthesis |
| `AnswererAgent` | Question answering and response generation |
| `CoordinatorAgent` | Workflow orchestration and task management |






## Installation
 1. Clone the repository


```bash
  git clone https://github.com/yourusername/multi-agent-system.git
```

2. Navigate to project directory
```bash
  cd multi-agent-system
```
3. Run the system
```bash
  python main.py
```

## Usage/Examples

```javascript
from multi_agent_system import MultiAgentSystem

# Initialize and start the system
mas = MultiAgentSystem()
mas.start()

# Execute a task
result = mas.execute_task(
    "Research artificial intelligence trends",
    "What are the current trends in AI development?"
)

print(f"Status: {result['status']}")
print(f"Answer: {result['result']['answer']['answer']}")

# Stop the system
mas.stop()
}
```



## API Reference
### MultiAgentSystem
### Methods

## start()
```
Initializes and starts all agents in separate threads
```
## stop()
```
Gracefully shuts down all agents
```
## execute_task(description, question=None)
```
Executes a complete task workflow
Returns: Dictionary with status, result, duration, and steps
```
## get_system_status()
```
Returns system status including active agents and memory usage
```
## Roadmap

User Request → Coordinator → Planner → Research → Summarize → Answer → Response


## Deployment

### Requirements

* Python 3.7 or higher
* No external dependencies (uses standard library only)

### Production Considerations

* Implement proper logging
* Add error recovery mechanisms
* Configure appropriate timeouts
* Monitor agent performance
* Set up health checks
## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`` git checkout -b feature/new-feature``)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (``git commit -am 'Add new feature'``)
6. Push to the branch (``git push origin feature/new-feature``)
7. Create a Pull Request


## License
This project is licensed under the
[MIT](https://choosealicense.com/licenses/mit/) License.


## Authors

- [@Prasiddha10](https://github.com/Prasiddha10)


## Acknowledgements

* Multi-agent systems research community
* Python threading and concurrency patterns
* Open source contributors and maintainers

