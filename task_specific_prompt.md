The robot has been given this navigation task:

{task_prompt}

You are given recent camera frames ordered from oldest to newest.

Judge whether the newest frame shows that the robot has reached the final stopping position near the goal.

Return STOP if the newest frame has the target object AND it is taking up a majority of the view.

Return the required format and nothing else.
