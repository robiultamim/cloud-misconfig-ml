"""
feature_schema.py
─────────────────
Defines the complete 64-feature schema for cloud configuration manifests.
Every feature is a float (0.0 or 1.0 for booleans, normalised int for counts).
"""

FEATURE_NAMES = [
    # ── Kubernetes Security Context (Container Level) ──────────────────────
    "run_as_non_root_false",        # 1 if runAsNonRoot is False/absent (dangerous)
    "run_as_root_uid",              # 1 if runAsUser == 0
    "run_as_root_gid",              # 1 if runAsGroup == 0
    "privileged",                   # 1 if securityContext.privileged == true
    "allow_privilege_escalation",   # 1 if allowPrivilegeEscalation == true
    "read_only_root_fs_false",      # 1 if readOnlyRootFilesystem is absent/false
    "capabilities_add_all",         # 1 if capabilities.add contains ALL or SYS_ADMIN
    "capabilities_drop_missing",    # 1 if capabilities.drop is absent (risky)
    "seccomp_missing",              # 1 if no seccompProfile annotation/spec

    # ── Kubernetes Pod / Spec Level ────────────────────────────────────────
    "host_network",                 # 1 if hostNetwork: true
    "host_pid",                     # 1 if hostPID: true
    "host_ipc",                     # 1 if hostIPC: true
    "host_path_volume",             # 1 if any volume uses hostPath
    "service_account_default",      # 1 if serviceAccountName == "default" or absent
    "automount_sa_token",           # 1 if automountServiceAccountToken == true
    "image_latest_tag",             # 1 if container image uses :latest tag
    "image_no_digest",              # 1 if image has no @sha256 digest pinning
    "no_resource_limits",           # 1 if no CPU/memory limits set
    "no_resource_requests",         # 1 if no CPU/memory requests set
    "no_liveness_probe",            # 1 if no livenessProbe defined
    "no_readiness_probe",           # 1 if no readinessProbe defined
    "node_selector_missing",        # 1 if no nodeSelector/nodeAffinity

    # ── Kubernetes RBAC ────────────────────────────────────────────────────
    "cluster_admin_role",           # 1 if ClusterRole is cluster-admin
    "wildcard_verb",                # 1 if rules.verbs contains '*'
    "wildcard_resource",            # 1 if rules.resources contains '*'
    "wildcard_api_group",           # 1 if rules.apiGroups contains '*'
    "secrets_access",               # 1 if rules.resources contains 'secrets'
    "exec_access",                  # 1 if rules.verbs contains 'exec'
    "all_namespaces_binding",       # 1 if ClusterRoleBinding (not namespaced)

    # ── IAM / CloudFormation ───────────────────────────────────────────────
    "iam_wildcard_action",          # 1 if IAM Action contains '*'
    "iam_wildcard_resource",        # 1 if IAM Resource contains '*'
    "iam_allow_star",               # 1 if Effect:Allow + Action:* + Resource:*
    "iam_no_condition",             # 1 if Allow statement has no Condition block
    "iam_cross_account_trust",      # 1 if trust policy allows external AWS account
    "iam_passrole",                 # 1 if has iam:PassRole permission
    "iam_assume_role_star",         # 1 if sts:AssumeRole on Resource:*
    "iam_admin_policy",             # 1 if AdministratorAccess managed policy used
    "iam_inline_policy_admin",      # 1 if inline policy grants admin-equivalent

    # ── Cloud Storage / S3 ────────────────────────────────────────────────
    "s3_public_read_acl",           # 1 if ACL: public-read
    "s3_public_read_write_acl",     # 1 if ACL: public-read-write
    "s3_no_encryption",             # 1 if no ServerSideEncryptionConfiguration
    "s3_versioning_off",            # 1 if VersioningConfiguration.Status != Enabled
    "s3_no_logging",                # 1 if no LoggingConfiguration
    "s3_block_public_off",          # 1 if BlockPublicAcls or BlockPublicPolicy = false
    "s3_no_bucket_policy",          # 1 if no BucketPolicy resource linked
    "s3_http_only",                 # 1 if no HTTPS-only bucket policy condition

    # ── Serverless / Lambda (SAM) ──────────────────────────────────────────
    "lambda_no_vpc",                # 1 if Lambda has no VpcConfig
    "lambda_star_resource_policy",  # 1 if ResourcePolicy allows Principal: *
    "lambda_env_secrets",           # 1 if env vars contain keys like PASSWORD, SECRET
    "lambda_no_dlq",                # 1 if no DeadLetterQueue configured
    "lambda_admin_role",            # 1 if execution role is admin-equivalent
    "sam_no_auth",                  # 1 if API Gateway event has no Auth property

    # ── Structural / Metadata ──────────────────────────────────────────────
    "resource_type_k8s_pod",        # 1 if kind == Pod or Deployment
    "resource_type_rbac",           # 1 if kind is Role/ClusterRole/Binding
    "resource_type_iam",            # 1 if resource is IAM::Role or IAM::Policy
    "resource_type_s3",             # 1 if resource is S3::Bucket
    "resource_type_lambda",         # 1 if resource is Lambda::Function
    "namespace_default",            # 1 if namespace == "default"
    "has_labels",                   # 1 if metadata.labels exists and non-empty
    "has_annotations",              # 1 if metadata.annotations exists
    "has_inline_comment",           # 1 if YAML file has any # comment lines
    "nesting_depth_norm",           # nesting depth / 10 (normalised)
    "total_keys_norm",              # total key count / 100 (normalised)
    "multi_container",              # 1 if Pod has more than 1 container
]

assert len(FEATURE_NAMES) == 64, f"Feature count mismatch: {len(FEATURE_NAMES)}"

# Mapping: feature name → human-readable description (for SHAP reports)
FEATURE_DESCRIPTIONS = {
    "run_as_non_root_false":       "Container may run as root user (runAsNonRoot not enforced)",
    "privileged":                   "Container runs in privileged mode (full host access)",
    "allow_privilege_escalation":   "Privilege escalation from within container is permitted",
    "host_network":                 "Container shares host network namespace (bypasses isolation)",
    "host_pid":                     "Container shares host PID namespace",
    "host_path_volume":             "Container mounts host filesystem path (dangerous)",
    "image_latest_tag":             "Container image uses :latest tag (non-reproducible, risk of supply chain attack)",
    "cluster_admin_role":           "ClusterRole grants cluster-admin privileges",
    "wildcard_verb":                "RBAC rule grants all verbs (*) on resources",
    "iam_wildcard_action":          "IAM policy grants all actions (*) — full account access",
    "iam_allow_star":               "IAM statement: Allow all actions on all resources",
    "s3_public_read_acl":           "S3 bucket is publicly readable by anyone on the internet",
    "s3_no_encryption":             "S3 bucket has no server-side encryption configured",
    "lambda_star_resource_policy":  "Lambda function resource policy allows any Principal (*)",
}
