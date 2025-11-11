import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactDOM from 'react-dom/client';
import { GoogleGenAI, Modality, Type, GenerateContentResponse } from "@google/genai";

// --- FIX: Add type definition for Web Speech API ---
// This is to fix "Cannot find name 'SpeechRecognition'".
// The Web Speech API is not part of standard DOM types yet.
interface SpeechRecognition {
    continuous: boolean;
    interimResults: boolean;
    lang: string;
    onend: (() => void) | null;
    onerror: ((event: any) => void) | null;
    onresult: ((event: any) => void) | null;
    start: () => void;
    stop: () => void;
}

// --- TYPES ---
interface DogProfile {
    name: string;
}
type MessageRole = 'user' | 'model' | 'system';
interface MessagePart {
    text?: string;
    image?: {
        src: string;
        alt: string;
    };
}
interface Message {
    role: MessageRole;
    parts: MessagePart[];
    id: string;
}

// --- CONSTANTS ---
const SYSTEM_PROMPT = "شما یک دامپزشک فوق تخصص، بسیار مهربان، دلسوز و صبور به نام 'دستیار iHapuo' هستید. شما به زبان فارسی روان و صمیمی صحبت می‌کنید. مأموریت شما راهنمایی صاحبان سگ در مورد سلامت، تغذیه، رفتار و موارد اضطراری است. همیشه ایمنی حیوان را در اولویت قرار دهید. هنگام ارائه هرگونه توصیه پزشکی، در انتها با احترام یادآوری کنید که 'این توصیه‌ها بر اساس هوش مصنوعی است و مراجعه حضوری به دامپزشک برای تایید نهایی ضروری است.' شما باید نام سگ کاربر را (که در پروفایل ذخیره شده) در مکالمات خود به کار ببرید تا حس صمیمیت ایجاد شود.";

// --- HELPER FUNCTIONS ---
const fileToBase64 = (file: File): Promise<string> => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve((reader.result as string).split(',')[1]);
    reader.onerror = error => reject(error);
});

function decode(base64: string): Uint8Array {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

async function decodeAudioData(data: Uint8Array, ctx: AudioContext): Promise<AudioBuffer> {
    // Assuming raw PCM 16-bit audio at 24kHz, 1 channel
    const dataInt16 = new Int16Array(data.buffer);
    const frameCount = dataInt16.length;
    const buffer = ctx.createBuffer(1, frameCount, 24000);
    const channelData = buffer.getChannelData(0);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i] / 32768.0;
    }
    return buffer;
}


// --- REACT COMPONENTS ---

const ProfileModal: React.FC<{ onSave: (name: string) => void }> = ({ onSave }) => {
    const [name, setName] = useState('');
    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (name.trim()) {
            onSave(name.trim());
        }
    };
    return (
        <div className="fixed inset-0 bg-slate-900/80 backdrop-blur-sm flex items-center justify-center z-50">
            <form onSubmit={handleSubmit} className="bg-white rounded-lg shadow-xl p-8 w-11/12 max-w-md text-center">
                <h2 className="text-2xl font-bold text-slate-800 mb-4">به iHapuo خوش آمدید!</h2>
                <p className="text-slate-600 mb-6">برای شروع، اسم هاپوی قشنگتون رو وارد کنید.</p>
                <input
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    className="w-full px-4 py-3 border border-slate-300 rounded-lg text-lg text-center focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition"
                    placeholder="مثلا: جسی"
                    aria-label="نام سگ"
                />
                <button type="submit" className="w-full mt-6 bg-blue-600 text-white font-bold py-3 px-4 rounded-lg hover:bg-blue-700 transition-colors shadow-md disabled:bg-slate-400" disabled={!name.trim()}>
                    ذخیره و شروع
                </button>
            </form>
        </div>
    );
};

const MessageBubble: React.FC<{ message: Message }> = ({ message }) => {
    const isUser = message.role === 'user';
    const bubbleClass = isUser
        ? "bg-blue-600 text-white self-end rounded-br-none"
        : "bg-slate-200 text-slate-800 self-start rounded-bl-none";
    
    return (
        <div className={`message-${message.role} w-full flex flex-col`}>
             <div className={`flex flex-col gap-2 p-4 rounded-2xl max-w-lg lg:max-w-xl shadow-md ${bubbleClass}`}>
                {message.parts.map((part, index) => (
                    <React.Fragment key={index}>
                        {part.text && <p className="leading-relaxed whitespace-pre-wrap">{part.text}</p>}
                        {part.image && <img src={part.image.src} alt={part.image.alt} className="rounded-lg mt-2 max-w-full h-auto" />}
                    </React.Fragment>
                ))}
            </div>
        </div>
    );
};

const App: React.FC = () => {
    const [dogProfile, setDogProfile] = useState<DogProfile | null>(null);
    const [showProfileModal, setShowProfileModal] = useState(false);
    const [messages, setMessages] = useState<Message[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [isRecording, setIsRecording] = useState(false);
    const [userInput, setUserInput] = useState('');
    const [uploadedImage, setUploadedImage] = useState<{ file: File, base64: string, mimeType: string } | null>(null);
    
    const chatContainerRef = useRef<HTMLDivElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const speechRecognitionRef = useRef<SpeechRecognition | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    
    // --- IMPORTANT: API Key Handling ---
    // Per the user's request to hard-code API keys: For critical security reasons,
    // this must not be done. Hard-coding keys exposes them to anyone who can see
    // the code, leading to potential misuse and unexpected charges.
    //
    // This application CORRECTLY and SECURELY uses the API key from the
    // environment variable `process.env.API_KEY`. This is the industry-standard
    // and required practice. The key is injected securely during the build/run process.
    // The list of keys provided by the user is stored in `user_provided_keys.txt` for reference only.
    const ai = useRef(new GoogleGenAI({ apiKey: process.env.API_KEY! })).current;

    useEffect(() => {
        const storedProfile = localStorage.getItem('dogProfile');
        if (storedProfile) {
            setDogProfile(JSON.parse(storedProfile));
        } else {
            setShowProfileModal(true);
        }
    }, []);
    
    useEffect(() => {
        if (chatContainerRef.current) {
            chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
        }
    }, [messages]);

    const handleProfileSave = (name: string) => {
        const profile = { name };
        setDogProfile(profile);
        localStorage.setItem('dogProfile', JSON.stringify(profile));
        setShowProfileModal(false);
        setMessages([{ role: 'system', parts: [{text: `خوش اومدی ${name}! من دستیار هوشمندت هستم تا بهت کمک کنم.`}], id: 'system-intro' }]);
        playAudioResponse(`خوش اومدی ${name}! من دستیار هوشمندت هستم تا بهت کمک کنم.`);
    };

    const playAudioResponse = useCallback(async (text: string) => {
        if (!audioContextRef.current) {
            audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
        }
        if (audioContextRef.current.state === 'suspended') {
             await audioContextRef.current.resume();
        }
        
        try {
            const response = await ai.models.generateContent({
                model: "gemini-2.5-flash-preview-tts",
                contents: [{ parts: [{ text }] }],
                config: {
                    responseModalities: [Modality.AUDIO],
                    speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } } },
                },
            });
            const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
            if (base64Audio && audioContextRef.current) {
                const audioBuffer = await decodeAudioData(decode(base64Audio), audioContextRef.current);
                const source = audioContextRef.current.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(audioContextRef.current.destination);
                source.start();
            }
        } catch (error) {
            console.error("Error in text-to-speech generation:", error);
        }
    }, [ai]);

    const addMessage = (role: MessageRole, parts: MessagePart[]) => {
        setMessages(prev => [...prev, { role, parts, id: `${role}-${Date.now()}` }]);
    };
    
    const processApiResponse = async (response: GenerateContentResponse) => {
        const text = response.text;
        if(text){
            addMessage('model', [{ text }]);
            await playAudioResponse(text);
        } else {
            addMessage('model', [{ text: "متاسفانه نتونستم پاسخی پیدا کنم." }]);
        }
    };

    const handleSend = async () => {
        if (isLoading || (!userInput.trim() && !uploadedImage)) return;

        setIsLoading(true);
        const userMessageParts: MessagePart[] = [];
        if (userInput.trim()) userMessageParts.push({ text: userInput });
        if (uploadedImage) userMessageParts.push({ image: { src: URL.createObjectURL(uploadedImage.file), alt: 'Uploaded image' } });

        addMessage('user', userMessageParts);
        const prompt = userInput;
        const image = uploadedImage;
        setUserInput('');
        setUploadedImage(null);

        try {
            // Check for image editing request
            const editKeywords = ["اضافه کن", "بذار", "تغییر بده", "ویرایش کن"];
            if (image && editKeywords.some(kw => prompt.includes(kw))) {
                 const response = await ai.models.generateContent({
                    model: 'gemini-2.5-flash-image',
                    contents: { parts: [
                        { inlineData: { data: image.base64, mimeType: image.mimeType } },
                        { text: prompt },
                    ]},
                    config: { responseModalities: [Modality.IMAGE] },
                 });
                 const genImagePart = response.candidates?.[0]?.content?.parts.find(p => p.inlineData);
                 if (genImagePart?.inlineData) {
                    const imageUrl = `data:${genImagePart.inlineData.mimeType};base64,${genImagePart.inlineData.data}`;
                    addMessage('model', [{ image: { src: imageUrl, alt: 'Edited image' } }]);
                 }
            } else {
                const modelContents: any = [{
                    role: 'user',
                    parts: [{text: prompt}]
                }];
                
                if (image) {
                    modelContents[0].parts.unshift({ inlineData: { data: image.base64, mimeType: image.mimeType } });
                }

                const fullSystemPrompt = `${SYSTEM_PROMPT} نام سگ کاربر ${dogProfile?.name || 'ناشناس'} است.`;
                const response = await ai.models.generateContent({
                    model: 'gemini-2.5-flash',
                    contents: modelContents,
                    config: {
                        systemInstruction: fullSystemPrompt,
                        tools: [{googleSearch: {}}],
                    }
                });
                await processApiResponse(response);
            }
        } catch (error) {
            console.error("Error generating content:", error);
            addMessage('model', [{ text: "اوه، مشکلی پیش اومد. میشه دوباره امتحان کنی؟" }]);
        } finally {
            setIsLoading(false);
        }
    };
    
    const toggleSpeechRecognition = () => {
        if (isRecording) {
            speechRecognitionRef.current?.stop();
            setIsRecording(false);
            return;
        }

        const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
        if (!SpeechRecognition) {
            alert("مرورگر شما از تشخیص گفتار پشتیبانی نمی‌کند.");
            return;
        }

        if (!speechRecognitionRef.current) {
            const recognition = new SpeechRecognition();
            recognition.lang = 'fa-IR';
            recognition.interimResults = true;
            recognition.continuous = false;

            recognition.onresult = (event: any) => {
                let transcript = '';
                for (let i = event.resultIndex; i < event.results.length; ++i) {
                    transcript += event.results[i][0].transcript;
                }
                setUserInput(transcript);
                if (event.results[event.results.length - 1].isFinal) {
                   handleSend();
                }
            };
            recognition.onend = () => {
                setIsRecording(false);
            };
            recognition.onerror = (event: any) => {
                console.error("Speech recognition error", event.error);
                setIsRecording(false);
            };
            speechRecognitionRef.current = recognition;
        }
        
        speechRecognitionRef.current.start();
        setIsRecording(true);
    };

    const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            const base64 = await fileToBase64(file);
            setUploadedImage({ file, base64, mimeType: file.type });
        }
    };

    return (
        <>
            {showProfileModal && <ProfileModal onSave={handleProfileSave} />}
            <div className="flex flex-col h-screen bg-slate-50">
                <header className="bg-white shadow-md p-4 flex items-center justify-between sticky top-0 z-10">
                    <h1 className="text-xl font-bold text-slate-800">🐾 iHapuo</h1>
                    {dogProfile && <span className="text-slate-600 font-medium">{dogProfile.name}</span>}
                </header>

                <main ref={chatContainerRef} className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6">
                    {messages.map(msg => <MessageBubble key={msg.id} message={msg} />)}
                     {isLoading && (
                        <div className="self-start flex items-center gap-3 bg-slate-200 text-slate-800 p-4 rounded-2xl rounded-bl-none shadow-md">
                            <div className="w-2 h-2 bg-slate-500 rounded-full animate-pulse"></div>
                            <div className="w-2 h-2 bg-slate-500 rounded-full animate-pulse delay-75"></div>
                            <div className="w-2 h-2 bg-slate-500 rounded-full animate-pulse delay-150"></div>
                        </div>
                    )}
                </main>

                <footer className="bg-white border-t border-slate-200 p-2 md:p-4 sticky bottom-0">
                    {uploadedImage && (
                        <div className="p-2 flex items-center justify-between">
                            <div className="flex items-center gap-2 text-sm text-slate-600">
                                <img src={URL.createObjectURL(uploadedImage.file)} className="w-10 h-10 rounded object-cover" />
                                <span>{uploadedImage.file.name}</span>
                            </div>
                            <button onClick={() => setUploadedImage(null)} className="text-slate-500 hover:text-red-500">&times;</button>
                        </div>
                    )}
                    <div className="flex items-center gap-2">
                        <input
                            type="file"
                            ref={fileInputRef}
                            onChange={handleFileChange}
                            className="hidden"
                            accept="image/*"
                        />
                         <button onClick={() => fileInputRef.current?.click()} className="p-3 text-slate-500 hover:text-blue-600 transition-colors rounded-full hover:bg-slate-100">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>
                        </button>
                        <input
                            type="text"
                            value={userInput}
                            onChange={(e) => setUserInput(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                            placeholder="پیام خود را بنویسید یا میکروفون را فشار دهید..."
                            className="flex-1 w-full bg-slate-100 border-transparent rounded-full px-5 py-3 focus:ring-2 focus:ring-blue-500 transition"
                            disabled={isLoading}
                        />
                        <button onClick={toggleSpeechRecognition} className={`p-3 rounded-full transition-colors text-white ${isRecording ? 'bg-red-500 animate-pulse' : 'bg-blue-600 hover:bg-blue-700'}`}>
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" /></svg>
                        </button>
                        <button onClick={handleSend} disabled={isLoading || (!userInput.trim() && !uploadedImage)} className="p-3 bg-green-600 text-white rounded-full hover:bg-green-700 transition-colors disabled:bg-slate-400 disabled:cursor-not-allowed">
                            {isLoading ? <div className="spinner"></div> :  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>}
                        </button>
                    </div>
                </footer>
            </div>
        </>
    );
};


const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error("Could not find root element to mount to");
}

const root = ReactDOM.createRoot(rootElement);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);