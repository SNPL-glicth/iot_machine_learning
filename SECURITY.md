# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it privately.

**Do NOT** open a public issue for security vulnerabilities.

### How to Report

Send an email to the security team at: **zeninenterprise8@gmail.com**

Include the following information:
- Description of the vulnerability
- Steps to reproduce the vulnerability
- Potential impact of the vulnerability
- Any suggested fixes or mitigations (if known)

### What Happens Next

1. We will acknowledge receipt of your report within 48 hours
2. We will investigate the vulnerability and assess the severity
3. We will work on a fix and coordinate disclosure with you
4. Once a fix is available, we will release a security advisory
5. We will credit you for the discovery (if you wish)

### Security Best Practices

When using the ZENIN ML Engine:

- Keep dependencies updated
- Use environment variables for sensitive configuration (see `.env.example`)
- Do not commit `.env` files to version control
- Use HTTPS for all API communications
- Implement proper authentication and authorization in production
- Regularly review and update API keys and credentials
- Monitor logs for suspicious activity
- Keep the system and its dependencies patched

### Dependency Security

This project uses:
- `requirements.txt` for production dependencies
- `pyproject.toml` for development dependencies

We regularly update dependencies to address security vulnerabilities. Use:
```bash
pip install --upgrade -r requirements.txt
```

### Environment Variables

Sensitive configuration should be stored in environment variables. See `.env.example` for the required variables.

**Never commit actual `.env` files to the repository.**

### API Security

- All API endpoints require API key authentication
- Rate limiting is implemented to prevent abuse
- Input validation is performed on all endpoints
- SQL injection protection via parameterized queries
- CORS configuration should be set appropriately for your deployment

### Data Protection

- Personal data should be encrypted at rest and in transit
- Follow GDPR and other applicable data protection regulations
- Implement proper access controls for sensitive data
- Regular security audits are recommended

## Security Advisories

We will publish security advisories for significant vulnerabilities. Subscribe to the repository releases to be notified.

## Contact

For general security questions, please open an issue in the repository.
