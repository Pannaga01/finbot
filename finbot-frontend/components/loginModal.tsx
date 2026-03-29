"use client";
import { useState } from "react";
import { useAuth } from "../context/AuthContext";
import styles from './LoginModal.module.css';

const roles = [
    "employee",
    "hr",
    "finance",
    "marketing",
    "engineering",
    "c_level",
];

export default function LoginModal({ onClose }: { onClose: () => void }) {
    const [userId, setUserId] = useState("");
    const [role, setRole] = useState("employee");
    const { login } = useAuth();

    const handleLogin = () => {
        if (!userId) return alert("Enter user ID");
        login(userId, role);
        onClose();
    };

    return (
        <div className={styles.modalOverlay}>
            <div className={styles.modal}>
                <h2>Login</h2>

                <input
                    placeholder="User ID"
                    value={userId}
                    onChange={(e) => setUserId(e.target.value)}
                />

                <select value={role} onChange={(e) => setRole(e.target.value)}>
                    {roles.map((r) => (
                        <option key={r}>{r}</option>
                    ))}
                </select>

                <button onClick={handleLogin}>Login</button>
                <button className={styles.secondary} onClick={onClose}>
                    Cancel
                </button>
            </div>
        </div>
    );
}