"use client";
import { useAuth } from "../context/AuthContext";
import { useState } from "react";
import MessageBubble from "./MessageBubble";
import { sendQuery } from "../lib/api";
import { Message } from "@/types";
import styles from "./ChatBot.module.css";

export default function ChatBox() {
    const { user, logout } = useAuth();

    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);

    // ✅ Replace hardcoded values
    const user_id = user?.user_id;
    const role = user?.role;

    const handleSend = async () => {
        if (!input.trim()) return;

        // ✅ Safety check (optional but recommended)
        if (!user_id || !role) {
            alert("User not logged in");
            return;
        }

        const newMessages: Message[] = [
            ...messages,
            { role: "user", content: input },
        ];

        setMessages(newMessages);
        setInput("");
        setLoading(true);

        try {
            const res = await sendQuery({
                user_id,
                role,
                query: input,
            });

            const botReply =
                res.answer ||
                res.block_reason ||
                "⚠️ Something went wrong";

            setMessages([
                ...newMessages,
                { role: "assistant", content: botReply },
            ]);
        } catch (err) {
            setMessages([
                ...newMessages,
                {
                    role: "assistant",
                    content: "❌ Failed to connect to server",
                },
            ]);
        }

        setLoading(false);
    };

    return (
        <div className={styles.chatbox}>
            <div className={styles.chatboxHeader}>
                <span>FinBot 💼</span>
                <button onClick={logout} className={styles.logoutBtn}>
                    Logout
                </button>
            </div>

            <div className={styles.chatboxMessages}>
                {messages.map((msg, i) => (
                    <MessageBubble key={i} message={msg} />
                ))}
                {loading && <p className={styles.thinking}>Thinking...</p>}
            </div>

            <div className={styles.chatboxInput}>
                <input
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Ask something..."
                />
                <button onClick={handleSend}>Send</button>
            </div>
        </div>
    );
}