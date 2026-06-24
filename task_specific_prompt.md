The robot has been given this navigation task:
{task_prompt}

Judge whether the robot has already reached the goal in the current camera view.

Estimate the distance to the image based on how big the image is in the overall view.
- if it is visible but still small, then return CONTINUE
- if it is visible and taking up a large amount of the image, then return STOP

Definition of taking up large amount of image: 
- the object/target of interest is one of the only things in view
- all other objects are either obscured or you cannot see

Definition of visible but still small in view:
- the objct/target is visible but you can still see many other irrelevant objects
- you cannot the object in detail, and it seems small compared to other irrelevant objects in view

Return the required format and nothing else.
