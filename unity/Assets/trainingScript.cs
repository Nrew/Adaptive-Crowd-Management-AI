using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using Unity.Transforms;

public class RollerAgent : Agent
{
    // Start is called before the first frame update

    Rigidbody rBody;

    //public Transform Target;
    public float forceMultiplier = 10;


    public Vector3 areacenter = new(-5f, 2f, 2f);
    public Vector2 areaSize = new Vector2(10f, 10f);
    private static bool targetAreaCreated = false;
    public GameObject targetAreaVisual;

    private static HashSet<RollerAgent> agentsInTargetArea = new HashSet<RollerAgent>();
    private bool hasReachedTargetArea = false;
    private static int totalAgents = 3;


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
        }

        if (isInTargetArea() && !hasReachedTargetArea)
        {
            hasReachedTargetArea = true;
            agentsInTargetArea.Add(this);
            SetReward(100.0f);
            this.rBody.linearVelocity = Vector3.zero;
            Destroy(this.GetComponent<Collider>());

            if (AllAgentsInTargetArea())
            {
                RollerAgent[] allAgents = Object.FindObjectsByType<RollerAgent>(FindObjectsSortMode.None);
                foreach (RollerAgent agent in allAgents)
                {
                    agent.EndEpisode();
                }
            }
        }
        // If it falls off the edge, end all episodes and reset
        else if (this.transform.localPosition.y < -1)
        {
            SetReward(-5f);
            RollerAgent[] allAgents = Object.FindObjectsByType<RollerAgent>(FindObjectsSortMode.None);
            foreach (RollerAgent agent in allAgents)
            {
                agent.EndEpisode();
            }
        }
        else if (!hasReachedTargetArea)
        {
            //float previousDistance = Vector3.Distance(this.transform.localPosition - rBody.linearVelocity * Time.fixedDeltaTime, areacenter);
            //float distanceReward = (previousDistance - distanceToTarget);
            //float distanceToAreaCenter = Vector3.Distance(this.transform.localPosition, areacenter);

            //AddReward(distanceReward * 0.001f);
            float velocityReward = rBody.linearVelocity.magnitude * 0.01f;
            AddReward(velocityReward);
            AddReward(-0.01f);

            //if (distanceToAreaCenter > 30f)
            //{
            //    AddReward(-0.01f);
            //}
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