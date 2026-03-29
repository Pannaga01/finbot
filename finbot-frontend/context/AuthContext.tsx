"use client";
import { createContext, useContext, useState, useEffect } from "react";

type User = {
    user_id: string;
    role: string;
} | null;

type AuthContextType = {
    user: User;
    login: (user_id: string, role: string) => void;
    logout: () => void;
};

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider = ({ children }: { children: React.ReactNode }) => {
    const [user, setUser] = useState<User>(null);

    useEffect(() => {
        const stored = localStorage.getItem("finbot_user");
        if (stored) setUser(JSON.parse(stored));
    }, []);

    const login = (user_id: string, role: string) => {
        const userData = { user_id, role };
        setUser(userData);
        localStorage.setItem("finbot_user", JSON.stringify(userData));
    };

    const logout = () => {
        setUser(null);
        localStorage.removeItem("finbot_user");
    };

    return (
        <AuthContext.Provider value={{ user, login, logout }}>
            {children}
        </AuthContext.Provider>

    );
};

export const useAuth = () => {
    const context = useContext(AuthContext);
    if (!context) throw new Error("useAuth must be used within AuthProvider");
    return context;
};