using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class MyAgent : Agent
{
    [SerializeField] private Transform targetTransform;
    [SerializeField] private float maxDistance = 10f;
    [SerializeField] private float goalThreshold = 1f;
    [SerializeField] private float moveSpeed = 5f;

    private Vector3 startingPosition;
    private Rigidbody rBody;

    void Start()
    {
        rBody = GetComponent<Rigidbody>();
        startingPosition = transform.position;

        // Configure rigidbody settings
        rBody.constraints = RigidbodyConstraints.FreezeRotation | RigidbodyConstraints.FreezePositionY;
        rBody.drag = 0.5f;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // 1-2: Agent position (x, z)
        sensor.AddObservation(transform.position.x / maxDistance);
        sensor.AddObservation(transform.position.z / maxDistance);

        // 3-4: Target position (x, z)
        sensor.AddObservation(targetTransform.position.x / maxDistance);
        sensor.AddObservation(targetTransform.position.z / maxDistance);

        // 5-6: Agent velocity (x, z)
        sensor.AddObservation(rBody.velocity.x / moveSpeed);
        sensor.AddObservation(rBody.velocity.z / moveSpeed);

        // 7: Distance to target (normalized)
        float distanceToTarget = Vector3.Distance(transform.position, targetTransform.position);
        sensor.AddObservation(distanceToTarget / maxDistance);

        // 8: Angle to target (normalized)
        Vector3 directionToTarget = (targetTransform.position - transform.position).normalized;
        float angleToTarget = Vector3.SignedAngle(transform.forward, directionToTarget, Vector3.up) / 180f;
        sensor.AddObservation(angleToTarget);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // Get continuous actions
        float moveX = actions.ContinuousActions[0];
        float moveZ = actions.ContinuousActions[1];

        // Apply force instead of direct position change
        Vector3 force = new Vector3(moveX, 0, moveZ).normalized * moveSpeed;
        rBody.AddForce(force);

        // Calculate reward based on current state
        float distanceToTarget = Vector3.Distance(transform.position, targetTransform.position);

        // Distance-based reward
        float normalizedDistance = distanceToTarget / maxDistance;
        AddReward(-0.001f * normalizedDistance); // Small negative reward based on distance

        // Check if reached goal
        if (distanceToTarget < goalThreshold)
        {
            AddReward(1.0f);
            EndEpisode();
        }
        // Check if failed
        else if (Failed())
        {
            AddReward(-1.0f);
            EndEpisode();
        }
    }

    private bool Failed()
    {
        float distanceFromStart = Vector3.Distance(transform.position, startingPosition);
        return distanceFromStart > maxDistance || transform.position.y < -1f;
    }

    public override void OnEpisodeBegin()
    {
        // Reset agent position and velocity
        rBody.velocity = Vector3.zero;
        rBody.angularVelocity = Vector3.zero;
        transform.position = startingPosition;

        // Randomly position target within reasonable bounds
        float randomX = Random.Range(-maxDistance / 2f, maxDistance / 2f);
        float randomZ = Random.Range(-maxDistance / 2f, maxDistance / 2f);
        targetTransform.position = new Vector3(randomX, targetTransform.position.y, randomZ);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActions = actionsOut.ContinuousActions;
        continuousActions[0] = Input.GetAxisRaw("Horizontal");
        continuousActions[1] = Input.GetAxisRaw("Vertical");
    }
}