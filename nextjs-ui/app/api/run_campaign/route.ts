import { NextRequest, NextResponse } from 'next/server';

const DEFAULT_BACKEND_URL = 'http://127.0.0.1:8000';

function getBackendBaseUrl() {
  return (
    process.env.POPPER_BACKEND_URL ||
    process.env.NEXT_PUBLIC_POPPER_BACKEND_URL ||
    DEFAULT_BACKEND_URL
  ).replace(/\/+$/, '');
}

export async function POST(request: NextRequest) {
  const backendUrl = `${getBackendBaseUrl()}/api/run_campaign`;

  try {
    const body = await request.json();
    const response = await fetch(backendUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
      cache: 'no-store',
    });

    const text = await response.text();
    const contentType = response.headers.get('content-type') || 'application/json';

    return new NextResponse(text, {
      status: response.status,
      headers: {
        'Content-Type': contentType,
      },
    });
  } catch (error) {
    const message =
      error instanceof Error
        ? error.message
        : 'Unable to reach the backend validation API.';

    return NextResponse.json(
      {
        detail:
          `Could not reach the backend at ${backendUrl}. ` +
          `This is a proxy/backend failure rather than an Axios browser timeout. ` +
          `Start the FastAPI server on port 8000 or set POPPER_BACKEND_URL.`,
        error: message,
      },
      { status: 502 },
    );
  }
}
