import { useState } from 'react';
import { Copy, Check, Maximize2, Minimize2 } from 'lucide-react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus, coldarkDark, atomDark, tomorrow, dracula } from 'react-syntax-highlighter/dist/esm/styles/prism';

const CodeBlock = ({ code, language = 'python' }) => {
  const [copied, setCopied] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [theme, setTheme] = useState('vscDarkPlus');

  const themes = {
    vscDarkPlus: vscDarkPlus,
    coldarkDark: coldarkDark,
    atomDark: atomDark,
    tomorrow: tomorrow,
    dracula: dracula
  };

  const themeNames = {
    vscDarkPlus: 'VS Code Dark',
    coldarkDark: 'Coldark Dark',
    atomDark: 'Atom Dark',
    tomorrow: 'Tomorrow Night',
    dracula: 'Dracula'
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const toggleExpand = () => {
    setIsExpanded(!isExpanded);
  };

  return (
    <div className="relative bg-[#1e1e1e] rounded-xl overflow-hidden shadow-2xl border-2 border-gray-700 hover:border-gray-600 transition-all duration-300">
      {/* Top border accent */}
      <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500"></div>
      
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-3 bg-gradient-to-r from-gray-800 via-gray-850 to-gray-900 border-b-2 border-gray-700">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-red-500"></div>
            <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
            <div className="w-3 h-3 rounded-full bg-green-500"></div>
          </div>
          <div className="h-5 w-px bg-gray-600"></div>
          <span className="text-xs font-mono text-gray-300 uppercase tracking-wider font-semibold px-2 py-1 bg-gray-700 rounded border border-gray-600">
            {language}
          </span>
          <select
            value={theme}
            onChange={(e) => setTheme(e.target.value)}
            className="text-xs bg-gray-700 text-gray-300 border-2 border-gray-600 rounded-md px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 cursor-pointer hover:bg-gray-600 transition-colors"
          >
            {Object.keys(themes).map((themeName) => (
              <option key={themeName} value={themeName}>
                {themeNames[themeName]}
              </option>
            ))}
          </select>
        </div>
        
        <div className="flex items-center gap-2">
          <button
            onClick={toggleExpand}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-gray-700 hover:bg-gray-600 text-gray-200 rounded-md transition-all border border-gray-600 hover:border-gray-500"
            title={isExpanded ? "Collapse" : "Expand"}
          >
            {isExpanded ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
            <span className="hidden sm:inline">
              {isExpanded ? "Collapse" : "Expand"}
            </span>
          </button>
          <button
            onClick={handleCopy}
            className="flex items-center gap-2 px-4 py-1.5 text-xs bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-500 hover:to-blue-600 text-white rounded-md transition-all transform hover:scale-105 shadow-lg hover:shadow-blue-500/50 border border-blue-500"
          >
            {copied ? (
              <>
                <Check size={14} className="animate-bounce" />
                <span className="font-medium">Copied!</span>
              </>
            ) : (
              <>
                <Copy size={14} />
                <span className="font-medium">Copy Code</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Code Container */}
      <div 
        className={`relative overflow-auto transition-all duration-300 border-2 border-gray-800 ${
          isExpanded ? 'max-h-[800px]' : 'max-h-[500px]'
        }`}
        style={{ 
          scrollbarWidth: 'thin',
          scrollbarColor: '#4b5563 #1f2937'
        }}
      >
        <div className="border-l-4 border-blue-500/30">
          <SyntaxHighlighter
            language={language}
            style={themes[theme]}
            customStyle={{
              margin: 0,
              padding: '1.5rem',
              fontSize: '0.9rem',
              lineHeight: '1.6',
              borderRadius: 0,
              background: 'transparent',
              fontFamily: "'Fira Code', 'Consolas', 'Monaco', monospace",
            }}
            showLineNumbers={true}
            wrapLines={true}
            lineNumberStyle={{
              minWidth: '3.5em',
              paddingRight: '1.5em',
              color: '#6b7280',
              userSelect: 'none',
              borderRight: '2px solid #374151',
              marginRight: '1em',
              textAlign: 'right',
            }}
          >
            {code}
          </SyntaxHighlighter>
        </div>
      </div>

      {/* Gradient fade at bottom when not expanded */}
      {!isExpanded && (
        <div className="absolute bottom-0 left-0 right-0 h-24 bg-gradient-to-t from-[#1e1e1e] via-[#1e1e1e]/80 to-transparent pointer-events-none border-t border-gray-800"></div>
      )}
      
      {/* Bottom border accent */}
      <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-pink-500 via-purple-500 to-blue-500"></div>
    </div>
  );
};

export default CodeBlock;