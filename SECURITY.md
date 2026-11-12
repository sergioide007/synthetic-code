# Security policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 2.2.x   | Yes       |
| < 2.2   | No        |

## Reporting a vulnerability

Please **do not open a public issue** for a security problem. Report it privately through GitHub:
**Security → Report a vulnerability** on this repository (private vulnerability reporting).

Include the affected module and version, a minimal reproduction, and the impact you expect. You will get an
acknowledgement within 7 days. Fixes are released as a patch version and credited in the changelog unless you prefer
to stay anonymous.

## Scope

In scope: the `sintetico`, `sintetico_api` and `trazabilidad` packages, the `sintetico` command-line tool and the CI
configuration of this repository. Of particular interest:

* ways to alter `AgentAuditLogger` or evidence records without verification failing (beyond the documented limit:
  deleting the newest events is only detectable with an external anchor, see the README);
* inputs that make a guard, gate or reviewer **allow** what it should block (fail-open behaviour);
* resource exhaustion in the admission webhook or the API.

Out of scope: the deterministic `MockProvider`, the example regulatory profiles (they are examples, not legal
advice), and findings that require an attacker who already controls the signing key or the host.

## Hardening notes for adopters

* Load the audit key from `SINTETICO_AUDIT_KEY` (or your secret manager); never hardcode it.
* Store `AgentAuditLogger.head` outside the log and pass it to `verify(expected_head=...)`.
* Set `SINTETICO_API_KEY` before exposing the dashboard API beyond localhost.
* `SecurityScanner` is a cheap first filter, not a complete prompt-injection defense: combine it with a classifier
  and with tool-level permissions.
