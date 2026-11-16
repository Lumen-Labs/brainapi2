# Expert know-how gold production

Scored items are multi-hop questions over **prometheus-operator/runbooks** (Apache-2.0).

1. Parse each runbook's `## Mitigation` (or Response/Steps) and `## Diagnosis`/`## Meaning` first action-bearing sentence.
2. Pair two runbooks in the same component (or a runbook and a markdown-linked peer).
3. Gold `must_include` spans are action-bearing phrases unique to each hop document (not truncated mid-function-word). Answering requires both documents.
4. Original Northwind SRE/lab fixtures remain `ci_smoke`.

Committed gold was repaired so each hop span occurs in its hop document and not as an 8-word truncated prefix.

## Spot-check sample

### 1. `ek-mh-etcdGRPCRequestsSlow-etcdNoLeader-4`

- split: `test`
- gold method: `runbook_section_extraction_multi_hop`
- question: A etcd incident fires both etcdGRPCRequestsSlow and etcdNoLeader. What is the first mitigation guidance in the etcdGRPCRequestsSlow runbook, and what does the etcdNoLeader runbook say first under diagnosis/meaning?
- gold: `queries used to troubleshoot Verify the value of how slow the etcd gRPC requests are by / can happen if nodes from the cluster are orphaned - they were part of the cluster`
- must_include: `['queries used to troubleshoot Verify the value of how slow the etcd gRPC requests are by', 'can happen if nodes from the cluster are orphaned - they were part of the cluster']`

### 2. `ek-mh-NodeNetworkInterfaceFlapping-TargetDown-0`

- split: `test`
- gold method: `runbook_section_extraction_multi_hop`
- question: A general incident fires both NodeNetworkInterfaceFlapping and TargetDown. What is the first mitigation guidance in the NodeNetworkInterfaceFlapping runbook, and what does the TargetDown runbook say first under diagnosis/meaning?
- gold: `Check physical cables, check networking firewall rules and so / /targets page in Prometheus UI can be used to check the scrape error for the particular`
- must_include: `['Check physical cables, check networking firewall rules and so', '/targets page in Prometheus UI can be used to check the scrape error for the particular']`

### 3. `ek-mh-KubeAggregatedAPIDown-KubeCPUOvercommit-1`

- split: `test`
- gold method: `runbook_section_extraction_multi_hop`
- question: A kubernetes incident fires both KubeAggregatedAPIDown and KubeCPUOvercommit. What is the first mitigation guidance in the KubeAggregatedAPIDown runbook, and what does the KubeCPUOvercommit runbook say first under diagnosis/meaning?
- gold: `Impact From minor such as inability to see cluster metrics to more severe such as unable / Diagnosis - Check if CPU resource requests are adjusted to the app usage - Check if`
- must_include: `['Impact From minor such as inability to see cluster metrics to more severe such as unable', 'Diagnosis - Check if CPU resource requests are adjusted to the app usage - Check if']`

### 4. `ek-mh-KubeCPUQuotaOvercommit-KubeContainerWaiting-2`

- split: `test`
- gold method: `runbook_section_extraction_multi_hop`
- question: A kubernetes incident fires both KubeCPUQuotaOvercommit and KubeContainerWaiting. What is the first mitigation guidance in the KubeCPUQuotaOvercommit runbook, and what does the KubeContainerWaiting runbook say first under diagnosis/meaning?
- gold: `Diagnosis - Check if CPU resource requests are adjusted to the app usage - Check if / Diagnosis - Check pod events via kubectl -n $NAMESPACE describe pod $POD.`
- must_include: `['Diagnosis - Check if CPU resource requests are adjusted to the app usage - Check if', 'Diagnosis - Check pod events via kubectl -n $NAMESPACE describe pod $POD.']`

### 5. `ek-mh-KubeDaemonSetMisScheduled-KubeDaemonSetNotScheduled-0`

- split: `test`
- gold method: `runbook_section_extraction_multi_hop`
- question: A kubernetes incident fires both KubeDaemonSetMisScheduled and KubeDaemonSetNotScheduled. What is the first mitigation guidance in the KubeDaemonSetMisScheduled runbook, and what does the KubeDaemonSetNotScheduled runbook say first under diagnosis/meaning?
- gold: `Meaning A number of pods of daemonset are running where they are not supposed to run. / it was evicted by other more important pods - resources - maybe it tries to use`
- must_include: `['Meaning A number of pods of daemonset are running where they are not supposed to run.', 'it was evicted by other more important pods - resources - maybe it tries to use']`

### 6. `ek-mh-KubeDeploymentGenerationMismatch-KubeDeploymentReplicasMismatch-0`

- split: `test`
- gold method: `runbook_section_extraction_multi_hop`
- question: A kubernetes incident fires both KubeDeploymentGenerationMismatch and KubeDeploymentReplicasMismatch. What is the first mitigation guidance in the KubeDeploymentGenerationMismatch runbook, and what does the KubeDeploymentReplicasMismatch runbook say first under diagnosis/meaning?
- gold: `KubeDeploymentGenerationMismatch Meaning Deployment generation mismatch due to possible roll-back. / Diagnosis - Check deployment status via kubectl -n $NAMESPACE describe deployment $NAME.`
- must_include: `['KubeDeploymentGenerationMismatch Meaning Deployment generation mismatch due to possible roll-back.', 'Diagnosis - Check deployment status via kubectl -n $NAMESPACE describe deployment $NAME.']`

### 7. `ek-mh-KubeHpaReplicasMismatch-KubeJobCompletion-0`

- split: `test`
- gold method: `runbook_section_extraction_multi_hop`
- question: A kubernetes incident fires both KubeHpaReplicasMismatch and KubeJobCompletion. What is the first mitigation guidance in the KubeHpaReplicasMismatch runbook, and what does the KubeJobCompletion runbook say first under diagnosis/meaning?
- gold: `Diagnosis Check why HPA was unable to scale: - not enough nodes in the cluster - / - Possible issues with scheduling next Job Diagnosis - Check job via kubectl -n $NAMESPACE describe`
- must_include: `['Diagnosis Check why HPA was unable to scale: - not enough nodes in the cluster -', '- Possible issues with scheduling next Job Diagnosis - Check job via kubectl -n $NAMESPACE describe']`

### 8. `ek-mh-KubeMemoryOvercommit-KubeMemoryQuotaOvercommit-0`

- split: `test`
- gold method: `runbook_section_extraction_multi_hop`
- question: A kubernetes incident fires both KubeMemoryOvercommit and KubeMemoryQuotaOvercommit. What is the first mitigation guidance in the KubeMemoryOvercommit runbook, and what does the KubeMemoryQuotaOvercommit runbook say first under diagnosis/meaning?
- gold: `are available and not cordoned - Check if cluster-autoscaler has issues with adding new nodes Mitigation / are available and not cordoned - Check if cluster-autoscaler has issues with adding new nodes -`
- must_include: `['are available and not cordoned - Check if cluster-autoscaler has issues with adding new nodes Mitigation', 'are available and not cordoned - Check if cluster-autoscaler has issues with adding new nodes -']`

### 9. `ek-mh-KubeNodeReadinessFlapping-KubeNodeUnreachable-0`

- split: `test`
- gold method: `runbook_section_extraction_multi_hop`
- question: A kubernetes incident fires both KubeNodeReadinessFlapping and KubeNodeUnreachable. What is the first mitigation guidance in the KubeNodeReadinessFlapping runbook, and what does the KubeNodeUnreachable runbook say first under diagnosis/meaning?
- gold: `KubeNodeReadinessFlapping Meaning The readiness status of node has changed few times in the last 15 minutes. / Kubernetes node is unreachable and some workloads may be rescheduled. Impact The performance of the cluster`
- must_include: `['KubeNodeReadinessFlapping Meaning The readiness status of node has changed few times in the last 15 minutes.', 'Kubernetes node is unreachable and some workloads may be rescheduled. Impact The performance of the cluster']`

### 10. `ek-mh-KubePersistentVolumeFillingUp-KubePodCrashLooping-0`

- split: `test`
- gold method: `runbook_section_extraction_multi_hop`
- question: A kubernetes incident fires both KubePersistentVolumeFillingUp and KubePodCrashLooping. What is the first mitigation guidance in the KubePersistentVolumeFillingUp runbook, and what does the KubePodCrashLooping runbook say first under diagnosis/meaning?
- gold: `Impact Service degradation, switching to read only mode. / is in CrashLoop which means the app dies or is unresponsive and kubernetes tries to restart`
- must_include: `['Impact Service degradation, switching to read only mode.', 'is in CrashLoop which means the app dies or is unresponsive and kubernetes tries to restart']`

### 11. `ek-mh-KubeProxyDown-KubeQuotaExceeded-1`

- split: `test`
- gold method: `runbook_section_extraction_multi_hop`
- question: A kubernetes incident fires both KubeProxyDown and KubeQuotaExceeded. What is the first mitigation guidance in the KubeProxyDown runbook, and what does the KubeQuotaExceeded runbook say first under diagnosis/meaning?
- gold: `Diagnosis Check the status of the kube-proxy daemon sets in the kube-system namespace. / Diagnosis - Check resource usage for the namespace in given time span Mitigation - Review existing`
- must_include: `['Diagnosis Check the status of the kube-proxy daemon sets in the kube-system namespace.', 'Diagnosis - Check resource usage for the namespace in given time span Mitigation - Review existing']`

### 12. `ek-mh-KubeSchedulerDown-KubeStatefulSetGenerationMismatch-0`

- split: `test`
- gold method: `runbook_section_extraction_multi_hop`
- question: A kubernetes incident fires both KubeSchedulerDown and KubeStatefulSetGenerationMismatch. What is the first mitigation guidance in the KubeSchedulerDown runbook, and what does the KubeStatefulSetGenerationMismatch runbook say first under diagnosis/meaning?
- gold: `Mitigation See old CoreOS docs in Web Archive / KubeStatefulSetGenerationMismatch Meaning StatefulSet generation mismatch due to possible roll-back.`
- must_include: `['Mitigation See old CoreOS docs in Web Archive', 'KubeStatefulSetGenerationMismatch Meaning StatefulSet generation mismatch due to possible roll-back.']`
