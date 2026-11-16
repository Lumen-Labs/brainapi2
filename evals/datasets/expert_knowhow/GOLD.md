# Expert know-how gold production

Scored items are multi-hop questions over **prometheus-operator/runbooks** (Apache-2.0).

1. Parse each runbook's `## Mitigation` (or Response/Steps) and `## Diagnosis`/`## Meaning` first sentence.
2. Pair two runbooks in the same component (or a runbook and a markdown-linked peer).
3. Gold `must_include` spans are those extracted sentences (truncated). Answering requires both documents.
4. Original Northwind SRE/lab fixtures remain `ci_smoke`.

## Human review (sample)

- `ek-mh-etcdGRPCRequestsSlow-etcdNoLeader-4` — **pass as multi-hop source spans**. The etcdGRPCRequestsSlow runbook's Mitigation section starts with fragmented-state / slow-disk guidance; etcdNoLeader discusses control-plane nodes. Both `must_include` spans occur in those files. Gold is extractive, not a paraphrase; markdown flattening can glue a heading to the next sentence (including upstream wording).
- `ek-mh-NodeNetworkInterfaceFlapping-TargetDown-0` — **pass**. Flapping runbook says "Cordon and drain node…"; TargetDown points at the Prometheus `/targets` page.
- `ek-mh-KubeDaemonSetMisScheduled-KubeDaemonSetNotScheduled-0` — **pass**. Distinct first mitigation vs diagnosis sentences in two kubernetes runbooks.

Automated check: every test `must_include` span occurs in the concatenated runbook corpus. Answering a question requires combining two documents (`hops`).

Builder meta: `{"n_runbooks": 100, "n_candidate_questions": 332, "components": ["alertmanager", "etcd", "general", "kube-state-metrics", "kubernetes", "node", "prometheus", "prometheus-operator"]}`

## Spot-check sample

### 1. `ek-mh-etcdGRPCRequestsSlow-etcdNoLeader-4`

- split: `test`
- gold method: `runbook_section_extraction_multi_hop`
- question: A etcd incident fires both etcdGRPCRequestsSlow and etcdNoLeader. What is the first mitigation guidance in the etcdGRPCRequestsSlow runbook, and what does the etcdNoLeader runbook say first under diagnosis/meaning?
- gold: `Fragmented state In the case of slow fisk / Control plane nodes issue This can occur multiple`
- must_include: `['Fragmented state In the case of slow fisk', 'Control plane nodes issue This can occur multiple']`

### 2. `ek-mh-NodeNetworkInterfaceFlapping-TargetDown-0`

- split: `test`
- gold method: `runbook_section_extraction_multi_hop`
- question: A general incident fires both NodeNetworkInterfaceFlapping and TargetDown. What is the first mitigation guidance in the NodeNetworkInterfaceFlapping runbook, and what does the TargetDown runbook say first under diagnosis/meaning?
- gold: `Cordon and drain node to migrate apps from / /targets page in Prometheus UI can be used`
- must_include: `['Cordon and drain node to migrate apps from', '/targets page in Prometheus UI can be used']`

### 3. `ek-mh-KubeAggregatedAPIDown-KubeCPUOvercommit-1`

- split: `test`
- gold method: `runbook_section_extraction_multi_hop`
- question: A kubernetes incident fires both KubeAggregatedAPIDown and KubeCPUOvercommit. What is the first mitigation guidance in the KubeAggregatedAPIDown runbook, and what does the KubeCPUOvercommit runbook say first under diagnosis/meaning?
- gold: `TODO See APIServer aggregation / - Check if CPU resource requests are adjusted`
- must_include: `['TODO See APIServer aggregation', '- Check if CPU resource requests are adjusted']`

### 4. `ek-mh-KubeCPUQuotaOvercommit-KubeContainerWaiting-2`

- split: `test`
- gold method: `runbook_section_extraction_multi_hop`
- question: A kubernetes incident fires both KubeCPUQuotaOvercommit and KubeContainerWaiting. What is the first mitigation guidance in the KubeCPUQuotaOvercommit runbook, and what does the KubeContainerWaiting runbook say first under diagnosis/meaning?
- gold: `- Review existing quota for given namespace and / - Check pod events via kubectl -n $NAMESPACE`
- must_include: `['- Review existing quota for given namespace and', '- Check pod events via kubectl -n $NAMESPACE']`

### 5. `ek-mh-KubeDaemonSetMisScheduled-KubeDaemonSetNotScheduled-0`

- split: `test`
- gold method: `runbook_section_extraction_multi_hop`
- question: A kubernetes incident fires both KubeDaemonSetMisScheduled and KubeDaemonSetNotScheduled. What is the first mitigation guidance in the KubeDaemonSetMisScheduled runbook, and what does the KubeDaemonSetNotScheduled runbook say first under diagnosis/meaning?
- gold: `Update DaemonSet and apply change, delete pods manually. / Usually happens when specifying wrong pod taints/affinities or`
- must_include: `['Update DaemonSet and apply change, delete pods manually.', 'Usually happens when specifying wrong pod taints/affinities or']`

### 6. `ek-mh-KubeDeploymentGenerationMismatch-KubeDeploymentReplicasMismatch-0`

- split: `test`
- gold method: `runbook_section_extraction_multi_hop`
- question: A kubernetes incident fires both KubeDeploymentGenerationMismatch and KubeDeploymentReplicasMismatch. What is the first mitigation guidance in the KubeDeploymentGenerationMismatch runbook, and what does the KubeDeploymentReplicasMismatch runbook say first under diagnosis/meaning?
- gold: `Depending on the conditions usually adding new nodes / - Check deployment status via kubectl -n $NAMESPACE`
- must_include: `['Depending on the conditions usually adding new nodes', '- Check deployment status via kubectl -n $NAMESPACE']`

### 7. `ek-mh-KubeHpaReplicasMismatch-KubeJobCompletion-0`

- split: `test`
- gold method: `runbook_section_extraction_multi_hop`
- question: A kubernetes incident fires both KubeHpaReplicasMismatch and KubeJobCompletion. What is the first mitigation guidance in the KubeHpaReplicasMismatch runbook, and what does the KubeJobCompletion runbook say first under diagnosis/meaning?
- gold: `In case of cluster-autoscaler you may need to / - Check job via kubectl -n $NAMESPACE describe`
- must_include: `['In case of cluster-autoscaler you may need to', '- Check job via kubectl -n $NAMESPACE describe']`

### 8. `ek-mh-KubeMemoryOvercommit-KubeMemoryQuotaOvercommit-0`

- split: `test`
- gold method: `runbook_section_extraction_multi_hop`
- question: A kubernetes incident fires both KubeMemoryOvercommit and KubeMemoryQuotaOvercommit. What is the first mitigation guidance in the KubeMemoryOvercommit runbook, and what does the KubeMemoryQuotaOvercommit runbook say first under diagnosis/meaning?
- gold: `- Add more nodes to the cluster - / - Check if Memory resource requests are adjusted`
- must_include: `['- Add more nodes to the cluster -', '- Check if Memory resource requests are adjusted']`

### 9. `ek-mh-KubeNodeReadinessFlapping-KubeNodeUnreachable-0`

- split: `test`
- gold method: `runbook_section_extraction_multi_hop`
- question: A kubernetes incident fires both KubeNodeReadinessFlapping and KubeNodeUnreachable. What is the first mitigation guidance in the KubeNodeReadinessFlapping runbook, and what does the KubeNodeUnreachable runbook say first under diagnosis/meaning?
- gold: `In case of maintenance ensure to cordon and / The notification details should list the node that's`
- must_include: `['In case of maintenance ensure to cordon and', "The notification details should list the node that's"]`

### 10. `ek-mh-KubePersistentVolumeFillingUp-KubePodCrashLooping-0`

- split: `test`
- gold method: `runbook_section_extraction_multi_hop`
- question: A kubernetes incident fires both KubePersistentVolumeFillingUp and KubePodCrashLooping. What is the first mitigation guidance in the KubePersistentVolumeFillingUp runbook, and what does the KubePodCrashLooping runbook say first under diagnosis/meaning?
- gold: `Data retention Deleting no longer needed data is / - Check template via kubectl -n $NAMESPACE get`
- must_include: `['Data retention Deleting no longer needed data is', '- Check template via kubectl -n $NAMESPACE get']`

### 11. `ek-mh-KubeProxyDown-KubeQuotaExceeded-1`

- split: `test`
- gold method: `runbook_section_extraction_multi_hop`
- question: A kubernetes incident fires both KubeProxyDown and KubeQuotaExceeded. What is the first mitigation guidance in the KubeProxyDown runbook, and what does the KubeQuotaExceeded runbook say first under diagnosis/meaning?
- gold: `AWS EKS If you are running AWS EKS / - Check resource usage for the namespace in`
- must_include: `['AWS EKS If you are running AWS EKS', '- Check resource usage for the namespace in']`

### 12. `ek-mh-KubeSchedulerDown-KubeStatefulSetGenerationMismatch-0`

- split: `test`
- gold method: `runbook_section_extraction_multi_hop`
- question: A kubernetes incident fires both KubeSchedulerDown and KubeStatefulSetGenerationMismatch. What is the first mitigation guidance in the KubeSchedulerDown runbook, and what does the KubeStatefulSetGenerationMismatch runbook say first under diagnosis/meaning?
- gold: `See old CoreOS docs in Web Archive / See Kubernetes Docs - Failed Deployment which can`
- must_include: `['See old CoreOS docs in Web Archive', 'See Kubernetes Docs - Failed Deployment which can']`
