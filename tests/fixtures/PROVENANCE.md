# Fixture provenance (primary sources, not hand-written)

| File | Produced by | How to regenerate |
|------|-------------|-------------------|
| `real_opentofu_plan.json` | **OpenTofu v1.13.2** (`tofu show -json`), built-in `terraform_data` resources (create / update / delete) | apply a config with `terraform_data.ledger`, `terraform_data.queue`; change it; `tofu plan -out=p && tofu show -json p` |
| `k8s_admission_v1_fields.json` | Field names + required flags parsed from the official Go types `kubernetes/api` `admission/v1/types.go` (commit in the file) | `curl` the raw file, parse `json:"..."` struct tags |

AWS attribute names used by `devops.iac_review` (`instance_class`, `multi_az`, `identifier`, `tags`, `from_port`, `to_port`,
`protocol`, `cidr_blocks`, `ipv6_cidr_blocks`, `cidr_ipv4`, `cidr_ipv6`, `ip_protocol`; `-1` = all protocols/ports) were checked
against `hashicorp/terraform-provider-aws` `website/docs/r/{db_instance,security_group,security_group_rule,vpc_security_group_ingress_rule}`.
