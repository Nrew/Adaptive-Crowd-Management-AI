using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using Unity.Transforms;
using Unity.Entities.UniversalDelegates;

public class RollerAgent : Agent
{
    // Start is called before the first frame update

    Rigidbody rBody;

    //public Transform Target;
    public float forceMultiplier = 5f;


    public Vector3 areacenter = new(-5f, 2f, 2f);
    public Vector2 areaSize = new Vector2(10f, 10f);
    private static bool targetAreaCreated = false;
    public GameObject targetAreaVisual;

    private static HashSet<RollerAgent> agentsInTargetArea = new HashSet<RollerAgent>();
    private bool hasReachedTargetArea = false;
    private static int totalAgents = 11;

    public float rayDistance = 5f;
    public float rayAngles = 12;


    void Start()
    {
        rBody = GetComponent<Rigidbody>();
        areacenter = new(-5f, 2f, 2f);
        if (!targetAreaCreated)
        {
            CreateTargetArea();
            targetAreaCreated = true;
        }
    }
    public override void OnEpisodeBegin()
    {
        this.rBody.angularVelocity = Vector3.zero;
        this.rBody.linearVelocity = Vector3.zero;
        this.transform.localPosition = new Vector3(Random.Range(-10, 0), 2f, Random.Range(30f, 40f));
        agentsInTargetArea = new HashSet<RollerAgent>();
        hasReachedTargetArea = false;

        if (this.transform.localPosition.y < 0)
        {
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.linearVelocity = Vector3.zero;
            this.transform.localPosition = new Vector3(Random.Range(-10, 0), 2f, Random.Range(30f, 40f));

        }

    }

    void CreateTargetArea()
    {
        Debug.Log($"Creating target area at position: {areacenter}");
        // Set the position and scale of the target area
        targetAreaVisual = GameObject.CreatePrimitive(PrimitiveType.Cube);
        targetAreaVisual.transform.localScale = new Vector3(areaSize.x, .1f, areaSize.y);

        targetAreaVisual.transform.position = areacenter;
        targetAreaVisual.transform.parent = transform.parent;
        // Make it green and transparent to easily identify it
        Renderer renderer = targetAreaVisual.GetComponent<Renderer>();
        renderer.material.color = Color.green;

        // Make it non-physical so that there are no collisions
        Destroy(targetAreaVisual.GetComponent<Collider>());
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(areacenter);
        sensor.AddObservation(this.transform.localPosition);

        sensor.AddObservation(rBody.linearVelocity.x);
        sensor.AddObservation(rBody.linearVelocity.z);
        sensor.AddObservation(hasReachedTargetArea);

        // detect walls
        float angleStep = 360f / rayAngles;
        for (float angle = 0; angle < 360; angle += angleStep)
        {
            Vector3 direction = Quaternion.Euler(0, angle, 0) * Vector3.forward;
            Ray ray = new Ray(this.transform.position, direction);
            RaycastHit hit;

            if (Physics.Raycast(ray, out hit, rayDistance))
            {
                sensor.AddObservation(hit.distance / rayDistance);  // Normalized distance to obstacle
            }
            else
            {
                sensor.AddObservation(1.0f);  // No obstacle detected
            }
        }
    }

    private bool isInTargetArea()
    {
        Vector3 agentPos = this.transform.position;
        float halfWidth = areaSize.x / 2;
        float halfLength = areaSize.y / 2;

        return agentPos.x >= (areacenter.x - halfWidth)
            && agentPos.z >= (areacenter.z - halfLength)
            && agentPos.x <= (areacenter.x + halfWidth)
            && agentPos.z <= (areacenter.z + halfLength);
    }
    private bool AllAgentsInTargetArea()
    {
        return agentsInTargetArea.Count >= totalAgents;
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        float distanceToTarget = Vector3.Distance(this.transform.localPosition, areacenter);
        if (!hasReachedTargetArea)
        {
            Vector3 controlSignal = Vector3.zero;
            controlSignal.x = actionBuffers.ContinuousActions[0];
            controlSignal.z = actionBuffers.ContinuousActions[1];
            rBody.AddForce(controlSignal * forceMultiplier);

            float movementReward = rBody.linearVelocity.magnitude * 0.002f;
            movementReward = Mathf.Min(movementReward, 0.001f);
            AddReward(movementReward);

            bool tooCloseToWall = false;
            bool closeEnough = false;
            int wallsDetected = 0;
            float minimumDistanceFromWall = 1.5f;
            //float closeToWall = 2.5f;

            for (float angle = 0; angle < 360f - float.Epsilon; angle += 360f / rayAngles) {
                Vector3 direction = Quaternion.Euler(0, angle, 0) * Vector3.forward;
                if (Physics.Raycast(this.transform.position, direction, out RaycastHit hit, minimumDistanceFromWall))
                {
                    tooCloseToWall = true;
                    wallsDetected++;
                }

            }
            if (tooCloseToWall)
            {
                AddReward(-0.0035f * wallsDetected);
            }
            else
            {
                AddReward(0.0035f);
            }

            AddReward(-0.0001f);

            // penalize being close to other agents
            //bool tooCloseToOthers = false;
            //RollerAgent[] agents = Object.FindObjectsByType<RollerAgent>(FindObjectsSortMode.None);
            //foreach (RollerAgent other in agents)
            //{
            //    if (other != this)
            //    {
            //        float agentDistance = Vector3.Distance(
            //            this.transform.localPosition,
            //            other.transform.localPosition
            //        );
            //        if (agentDistance < 2f)
            //        {
            //            tooCloseToOthers = true;
            //            //AddReward(-0.02f * (2f - agentDistance));
            //            break;
            //        }
            //    }
            //}

            //// reward for good positioning
            //if(!tooCloseToWall && !tooCloseToOthers)
            //{
            //    AddReward(0.03f);
            //}
            //// penalize for not moving
            //if (rBody.linearVelocity.magnitude < 0.1f)
            //{
            //    AddReward(-0.01f);
            //}
        }

        

        if (isInTargetArea() && !hasReachedTargetArea)
        {
            hasReachedTargetArea = true;
            agentsInTargetArea.Add(this);
            AddReward(100.0f);
            this.rBody.linearVelocity = Vector3.zero;
            //Destroy(this.GetComponent<Collider>());

            if (AllAgentsInTargetArea())
            {
                RollerAgent[] allAgents = Object.FindObjectsByType<RollerAgent>(FindObjectsSortMode.None);
                foreach (RollerAgent agent in allAgents)
                {
                    agent.EndEpisode();
                    agentsInTargetArea = new HashSet<RollerAgent>();
                }
            }
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetAxis("Horizontal");
        continuousActionsOut[1] = Input.GetAxis("Vertical");
    }

    void OnDestroy()
    {
        // Decrement total agent count when an agent is destroyed
        totalAgents--;
        agentsInTargetArea.Remove(this);

        // Clean up target area if this is the last agent
        if (targetAreaVisual != null && gameObject.scene.isLoaded)
        {
            RollerAgent[] remainingAgents = Object.FindObjectsByType<RollerAgent>(FindObjectsSortMode.None);
            if (remainingAgents.Length <= 1)
            {
                Destroy(targetAreaVisual);
                targetAreaCreated = false;
                agentsInTargetArea.Clear();
                totalAgents = 0;
            }
        }
    }

}