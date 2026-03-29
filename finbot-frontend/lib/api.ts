export async function sendQuery(data: {
    user_id: string;
    role: string;
    query: string;
}) {
    const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/query`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "x-session-id": data.user_id,
        },
        body: JSON.stringify(data),
    });

    return res.json();
}