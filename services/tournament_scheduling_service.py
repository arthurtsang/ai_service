"""
Tournament Scheduling Service for Kung Fu Tournament
Uses Mistral LLM to generate tournament schedules
"""

import logging
from typing import List, Dict, Any, Optional
import json
import asyncio
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.json_parser import extract_json_from_markdown

logger = logging.getLogger(__name__)

class TournamentSchedulingService:
    """LLM-based tournament scheduling service using Mistral"""
    
    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)
    
    async def generate_schedule(self, tournament_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate tournament schedule using Mistral LLM
        
        Args:
            tournament_data: Dictionary containing TournamentTimeTable data:
                - planning_events: List of events to schedule
                - rings: Available rings
                - head_judges: Available head judges
                - side_judges: Available side judges
                - helpers: Available helpers
                - dummies: Available dummies
                - time_slots: Available time slots
        
        Returns:
            List of scheduled PlanningEvent objects
        """
        try:
            self.logger.info(f"Generating schedule for {len(tournament_data.get('planning_events', []))} events using LLM")
            
            # Validate input data structure
            self._validate_tournament_data(tournament_data)
            
            # Create LLM prompt
            prompt = self._create_scheduling_prompt(tournament_data)
            
            # Get LLM response
            response = await self._get_llm_response(prompt)
            
            # Parse the response
            scheduled_events = self._parse_schedule_response(response)
            
            self.logger.info(f"Successfully generated schedule with {len(scheduled_events)} events")
            return scheduled_events
            
        except Exception as e:
            self.logger.error(f"Error generating schedule: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _validate_tournament_data(self, tournament_data: Dict[str, Any]):
        """Validate the tournament data structure"""
        required_keys = ['planning_events', 'rings', 'head_judges', 'side_judges', 'helpers', 'dummies', 'time_slots']
        
        for key in required_keys:
            if key not in tournament_data:
                raise ValueError(f"Missing required key: {key}")
        
        if not isinstance(tournament_data['planning_events'], list):
            raise ValueError("planning_events must be a list")
        
        if not isinstance(tournament_data['rings'], list):
            raise ValueError("rings must be a list")
        
        if not isinstance(tournament_data['time_slots'], list):
            raise ValueError("time_slots must be a list")
    
    def _create_scheduling_prompt(self, tournament_data: Dict[str, Any]) -> str:
        """Create a comprehensive prompt for the LLM to generate a tournament schedule"""
        
        prompt = f"""You are an expert tournament scheduler for Kung Fu competitions. Your task is to create an optimal schedule for a tournament by assigning resources to events while respecting all constraints.

**CRITICAL: You must respond with ONLY valid JSON. Do not include any explanations, code, or other text. Start your response with [ and end with ].**

## TOURNAMENT DATA

### Events to Schedule:
{json.dumps(tournament_data['planning_events'], indent=2)}

### Available Resources:
**Rings:**
{json.dumps(tournament_data['rings'], indent=2)}

**Head Judges:**
{json.dumps(tournament_data['head_judges'], indent=2)}

**Side Judges:**
{json.dumps(tournament_data['side_judges'], indent=2)}

**Helpers:**
{json.dumps(tournament_data['helpers'], indent=2)}

**Dummies:**
{json.dumps(tournament_data['dummies'], indent=2)}

**Time Slots:**
{json.dumps(tournament_data['time_slots'], indent=2)}

## SCHEDULING CONSTRAINTS

### HARD CONSTRAINTS (Must be satisfied):
1. **No Double Booking**: No person (judge, helper, dummy) can be assigned to overlapping events
2. **Resource Availability**: All assigned resources must be available for their assigned time slot
3. **Judge Qualifications**: Judges must be qualified for the event type, age group, belt rank, sex, and tai chi level
4. **Required Resources**: Events must have the minimum required number of side judges, helpers, and dummies
5. **Ring Suitability**: Rings must be suitable for the event type
6. **Demo Events**: Demo events don't need judges, helpers, or dummies (use null assignments)

### SOFT CONSTRAINTS (Optimize for):
1. **Time Preferences**: Schedule demo events and tai chi events at preferred times
2. **Age Group Blocking**: Keep same age groups together when possible
3. **Ring Consistency**: Minimize ring changes for the same age group
4. **Schedule Stability**: Minimize changes from original assignments if any

## ASSIGNMENT RULES

### For Regular Events:
- Assign exactly 1 head judge (must be qualified)
- Assign required_side_judges to max_side_judges side judges (must be qualified)
- Assign required_helpers to max_helpers helpers
- Assign required_dummies to max_dummies dummies
- Assign 1 ring (must be suitable for event type)
- Assign 1 time slot (must not conflict with other assignments)

### For Demo Events:
- Head judge: null (id: -1, name: "NullJudge")
- Side judges: all null (id: -1, name: "NullJudge")
- Helpers: all null (id: -1, name: "NullHelper")
- Dummies: all null (id: -1, name: "NullDummy")
- Ring: assign suitable ring
- Time slot: assign available time slot

## OUTPUT FORMAT

Return a JSON array of scheduled events. Each event should have this exact structure:

```json
[
  {{
    "id": "event_1",
    "event": {{
      "id": 1,
      "name": "Event Name",
      "event_type": "forms",
      "age_group": "children",
      "belt_rank": "white",
      "sex": "mixed",
      "tai_chi_level": null,
      "is_demo": false,
      "duration_minutes": 15
    }},
    "head_judge": {{
      "id": 1,
      "name": "Judge Name",
      "is_head_judge": true,
      "qualifications": {{}}
    }},
    "side_judge_1": {{
      "id": 2,
      "name": "Side Judge 1",
      "is_head_judge": false,
      "qualifications": {{}}
    }},
    "side_judge_2": {{
      "id": 3,
      "name": "Side Judge 2",
      "is_head_judge": false,
      "qualifications": {{}}
    }},
    "side_judge_3": {{
      "id": -1,
      "name": "NullJudge",
      "is_head_judge": false,
      "qualifications": {{}}
    }},
    "side_judge_4": {{
      "id": -1,
      "name": "NullJudge",
      "is_head_judge": false,
      "qualifications": {{}}
    }},
    "ring": {{
      "id": 1,
      "name": "Ring A",
      "suitable_for": ["forms", "sparring"]
    }},
    "start_time_slot": {{
      "id": 1,
      "start_time": "09:00",
      "end_time": "09:15",
      "day": 1
    }},
    "dummy_1": {{
      "id": 1,
      "name": "Dummy One"
    }},
    "dummy_2": {{
      "id": -1,
      "name": "NullDummy"
    }},
    "helper_1": {{
      "id": 1,
      "name": "Helper One"
    }},
    "helper_2": {{
      "id": -1,
      "name": "NullHelper"
    }}
  }}
]
```

## IMPORTANT NOTES:
- Use the exact IDs and names from the input data
- For null assignments, use id: -1 and the appropriate null name
- Ensure no resource conflicts (same person/ring/time slot used twice)
- Prioritize demo events for scheduling
- Return ONLY the JSON array, no other text

Generate the optimal tournament schedule now:"""

        return prompt
    
    async def _get_llm_response(self, prompt: str) -> str:
        """Get response from the LLM model"""
        try:
            if self.model is None or self.tokenizer is None:
                # Mock response for testing when model is not available
                self.logger.warning("LLM model not available, returning mock response")
                return self._get_mock_response()
            
            # Import torch here to avoid issues if not available
            import torch
            
            # Tokenize the input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4000)
            
            # Move inputs to the same device as the model
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=2000,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (remove the prompt)
            if prompt in response:
                response = response.split(prompt)[-1].strip()
            
            self.logger.info(f"LLM generated response: {response[:500]}...")
            return response
            
        except Exception as e:
            self.logger.error(f"Error getting LLM response: {str(e)}")
            # Return mock response on error
            return self._get_mock_response()
    
    def _get_mock_response(self) -> str:
        """Return a mock response for testing"""
        return '''[
  {
    "id": "event_1",
    "event": {
      "id": 1,
      "name": "Beginner Forms - Children",
      "event_type": "forms",
      "age_group": "children",
      "belt_rank": "white",
      "sex": "mixed",
      "tai_chi_level": null,
      "is_demo": false,
      "duration_minutes": 15
    },
    "head_judge": {
      "id": 1,
      "name": "Master Chen",
      "is_head_judge": true,
      "qualifications": {}
    },
    "side_judge_1": {
      "id": 3,
      "name": "Judge Wang",
      "is_head_judge": false,
      "qualifications": {}
    },
    "side_judge_2": {
      "id": 4,
      "name": "Judge Zhang",
      "is_head_judge": false,
      "qualifications": {}
    },
    "side_judge_3": {
      "id": -1,
      "name": "NullJudge",
      "is_head_judge": false,
      "qualifications": {}
    },
    "side_judge_4": {
      "id": -1,
      "name": "NullJudge",
      "is_head_judge": false,
      "qualifications": {}
    },
    "ring": {
      "id": 1,
      "name": "Ring A",
      "suitable_for": ["forms", "sparring", "demo"]
    },
    "start_time_slot": {
      "id": 1,
      "start_time": "09:00",
      "end_time": "09:15",
      "day": 1
    },
    "dummy_1": {
      "id": -1,
      "name": "NullDummy"
    },
    "dummy_2": {
      "id": -1,
      "name": "NullDummy"
    },
    "helper_1": {
      "id": 1,
      "name": "Helper One"
    },
    "helper_2": {
      "id": -1,
      "name": "NullHelper"
    }
  },
  {
    "id": "event_2",
    "event": {
      "id": 2,
      "name": "Advanced Sparring - Adults",
      "event_type": "sparring",
      "age_group": "adults",
      "belt_rank": "black",
      "sex": "mixed",
      "tai_chi_level": null,
      "is_demo": false,
      "duration_minutes": 20
    },
    "head_judge": {
      "id": 2,
      "name": "Master Lee",
      "is_head_judge": true,
      "qualifications": {}
    },
    "side_judge_1": {
      "id": 3,
      "name": "Judge Wang",
      "is_head_judge": false,
      "qualifications": {}
    },
    "side_judge_2": {
      "id": 4,
      "name": "Judge Zhang",
      "is_head_judge": false,
      "qualifications": {}
    },
    "side_judge_3": {
      "id": 5,
      "name": "Judge Liu",
      "is_head_judge": false,
      "qualifications": {}
    },
    "side_judge_4": {
      "id": -1,
      "name": "NullJudge",
      "is_head_judge": false,
      "qualifications": {}
    },
    "ring": {
      "id": 2,
      "name": "Ring B",
      "suitable_for": ["forms", "sparring"]
    },
    "start_time_slot": {
      "id": 2,
      "start_time": "09:15",
      "end_time": "09:30",
      "day": 1
    },
    "dummy_1": {
      "id": 1,
      "name": "Dummy One"
    },
    "dummy_2": {
      "id": 2,
      "name": "Dummy Two"
    },
    "helper_1": {
      "id": 1,
      "name": "Helper One"
    },
    "helper_2": {
      "id": 2,
      "name": "Helper Two"
    }
  },
  {
    "id": "event_3",
    "event": {
      "id": 3,
      "name": "Demo Event",
      "event_type": "demo",
      "age_group": "mixed",
      "belt_rank": "mixed",
      "sex": "mixed",
      "tai_chi_level": null,
      "is_demo": true,
      "duration_minutes": 10
    },
    "head_judge": {
      "id": -1,
      "name": "NullJudge",
      "is_head_judge": false,
      "qualifications": {}
    },
    "side_judge_1": {
      "id": -1,
      "name": "NullJudge",
      "is_head_judge": false,
      "qualifications": {}
    },
    "side_judge_2": {
      "id": -1,
      "name": "NullJudge",
      "is_head_judge": false,
      "qualifications": {}
    },
    "side_judge_3": {
      "id": -1,
      "name": "NullJudge",
      "is_head_judge": false,
      "qualifications": {}
    },
    "side_judge_4": {
      "id": -1,
      "name": "NullJudge",
      "is_head_judge": false,
      "qualifications": {}
    },
    "ring": {
      "id": 1,
      "name": "Ring A",
      "suitable_for": ["forms", "sparring", "demo"]
    },
    "start_time_slot": {
      "id": 3,
      "start_time": "09:30",
      "end_time": "09:45",
      "day": 1
    },
    "dummy_1": {
      "id": -1,
      "name": "NullDummy"
    },
    "dummy_2": {
      "id": -1,
      "name": "NullDummy"
    },
    "helper_1": {
      "id": -1,
      "name": "NullHelper"
    },
    "helper_2": {
      "id": -1,
      "name": "NullHelper"
    }
  }
]'''
    
    def _parse_schedule_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse the LLM response and extract the scheduled events"""
        try:
            # Extract JSON from the response (in case it's wrapped in markdown)
            json_str = extract_json_from_markdown(response)
            
            # Parse the JSON
            if isinstance(json_str, str):
                self.logger.info(f"Extracted JSON string: {json_str[:200]}...")
                scheduled_events = json.loads(json_str)
            elif isinstance(json_str, dict):
                # If it's already a dict, use it directly
                scheduled_events = json_str if isinstance(json_str, list) else [json_str]
            elif isinstance(json_str, list):
                scheduled_events = json_str
            else:
                self.logger.error(f"Unexpected json_str type: {type(json_str)}")
                raise ValueError(f"Unexpected JSON type: {type(json_str)}")
            
            self.logger.info(f"Parsed scheduled_events: {scheduled_events}")
            
            # Validate the response structure
            if not isinstance(scheduled_events, list):
                raise ValueError("Response must be a list of events")
            
            # Validate each event has required fields
            self.logger.info(f"Validating {len(scheduled_events)} events")
            for i, event in enumerate(scheduled_events):
                self.logger.info(f"Validating event {i}: {event.get('id', 'NO_ID')}")
                self.logger.info(f"Event {i} keys: {list(event.keys())}")
                required_fields = ['id', 'event', 'head_judge', 'ring', 'start_time_slot']
                for field in required_fields:
                    if field not in event:
                        self.logger.error(f"Event {i} missing required field: {field}")
                        self.logger.error(f"Event keys: {list(event.keys())}")
                        raise ValueError(f"Event missing required field: {field}")
            
            return scheduled_events
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {str(e)}")
            self.logger.error(f"Response was: {response}")
            raise ValueError(f"Invalid JSON response: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error parsing schedule response: {str(e)}")
            raise
