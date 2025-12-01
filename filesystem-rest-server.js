#!/usr/bin/env node

/**
 * Filesystem MCP REST Server
 * 
 * This script creates a REST API wrapper around the MCP filesystem server.
 * It allows HTTP clients to interact with the filesystem MCP service.
 * 
 * Usage:
 *   node filesystem-rest-server.js [--port PORT] [--directory DIR]
 * 
 * Environment Variables:
 *   MCP_SERVER_PORT - Port to listen on (default: 8001)
 *   MCP_ALLOWED_DIRECTORY - Directory to allow filesystem operations (required)
 */

const http = require('http');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

// Parse command line arguments
const args = process.argv.slice(2);
let port = process.env.MCP_SERVER_PORT || 8001;
let allowedDirectory = process.env.MCP_ALLOWED_DIRECTORY;

for (let i = 0; i < args.length; i++) {
  if (args[i] === '--port' && i + 1 < args.length) {
    port = parseInt(args[i + 1]);
    i++;
  } else if (args[i] === '--directory' && i + 1 < args.length) {
    allowedDirectory = args[i + 1];
    i++;
  }
}

if (!allowedDirectory) {
  console.error('Error: MCP_ALLOWED_DIRECTORY environment variable or --directory argument is required');
  process.exit(1);
}

console.log(`Starting Filesystem MCP REST Server...`);
console.log(`Port: ${port}`);
console.log(`Allowed Directory: ${allowedDirectory}`);

// MCP process management
let mcpProcess = null;
let currentAllowedDirectory = allowedDirectory;
let isInitializing = false;
let isReady = false;

// JSON-RPC message handling
let messageBuffer = '';
let responseHandlers = new Map();
let messageIdCounter = 1;

// Function to start/restart MCP process
async function startMCPProcess(directory) {
  isInitializing = true;
  isReady = false;
  
  // Kill existing process if any
  if (mcpProcess) {
    console.log('Stopping existing MCP process...');
    const oldProcess = mcpProcess;
    mcpProcess = null;  // Clear reference immediately
    
    // Remove all listeners to prevent interference
    oldProcess.removeAllListeners();
    
    // Kill and wait for it to exit
    oldProcess.kill('SIGTERM');
    
    // Wait a bit for process to fully exit
    await new Promise(resolve => setTimeout(resolve, 500));
    
    // Clear response handlers
    responseHandlers.clear();
  }
  
  // Verify directory exists (Python creates it before calling /configure)
  if (!fs.existsSync(directory)) {
    isInitializing = false;
    isReady = false;
    throw new Error(`Directory does not exist: ${directory}`);
  }
  
  console.log(`Starting MCP filesystem server with directory: ${directory}`);
  currentAllowedDirectory = directory;
  
  mcpProcess = spawn('npx', [
    '-y',
    '@modelcontextprotocol/server-filesystem',
    directory
  ], {
    stdio: ['pipe', 'pipe', 'pipe']  // Capture stderr too
  });
  
  // Capture stderr for debugging
  mcpProcess.stderr.on('data', (data) => {
    console.error('MCP stderr:', data.toString());
  });
  
  // Handle MCP process lifecycle
  mcpProcess.on('error', (error) => {
    console.error('Failed to start MCP server:', error);
    isInitializing = false;
    isReady = false;
  });

  mcpProcess.on('exit', (code, signal) => {
    console.log(`MCP server exited with code ${code}, signal ${signal}`);
    mcpProcess = null;
    isReady = false;
  });
  
  // Set up message handling
  setupMCPMessageHandling();
  
  // Wait for process to be ready to accept stdin
  // The MCP server prints startup messages before it's ready
  await new Promise(resolve => setTimeout(resolve, 2000));
  
  // Verify process is still running
  if (!mcpProcess || mcpProcess.exitCode !== null) {
    isInitializing = false;
    isReady = false;
    throw new Error('MCP process exited unexpectedly');
  }
  
  await initializeMCP();
  
  isInitializing = false;
  isReady = true;
}

function setupMCPMessageHandling() {
  if (!mcpProcess) return;
  
  messageBuffer = '';
  
  mcpProcess.stdout.on('data', (data) => {
    messageBuffer += data.toString();
    
    // Try to parse complete JSON-RPC messages
    const lines = messageBuffer.split('\n');
    messageBuffer = lines.pop(); // Keep incomplete line in buffer
    
    for (const line of lines) {
      if (line.trim()) {
        try {
          const message = JSON.parse(line);
          handleMCPResponse(message);
        } catch (e) {
          console.error('Failed to parse MCP response:', e);
        }
      }
    }
  });
}

function handleMCPResponse(message) {
  if (message.id && responseHandlers.has(message.id)) {
    const handler = responseHandlers.get(message.id);
    responseHandlers.delete(message.id);
    handler(message);
  }
}

function sendMCPRequest(method, params = {}) {
  return new Promise((resolve, reject) => {
    if (!mcpProcess || !mcpProcess.stdin) {
      reject(new Error('MCP process not ready'));
      return;
    }
    
    const id = messageIdCounter++;
    const request = {
      jsonrpc: '2.0',
      id,
      method,
      params
    };
    
    responseHandlers.set(id, (response) => {
      if (response.error) {
        reject(new Error(response.error.message || 'MCP request failed'));
      } else {
        resolve(response.result);
      }
    });
    
    try {
      mcpProcess.stdin.write(JSON.stringify(request) + '\n');
    } catch (error) {
      responseHandlers.delete(id);
      reject(new Error('Failed to write to MCP process: ' + error.message));
      return;
    }
    
    // Timeout after 30 seconds
    setTimeout(() => {
      if (responseHandlers.has(id)) {
        responseHandlers.delete(id);
        reject(new Error('Request timeout'));
      }
    }, 30000);
  });
}

// Initialize MCP session
async function initializeMCP() {
  try {
    await sendMCPRequest('initialize', {
      protocolVersion: '2024-11-05',
      capabilities: {},
      clientInfo: {
        name: 'filesystem-rest-server',
        version: '1.0.0'
      }
    });
    console.log('MCP server initialized successfully');
  } catch (error) {
    console.error('Failed to initialize MCP server:', error);
    throw error;
  }
}

// Start initial MCP process (async initialization)
(async () => {
  await startMCPProcess(allowedDirectory);
  console.log('Initial MCP process started and initialized');
})();

// Create HTTP server
const server = http.createServer(async (req, res) => {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  if (req.method === 'OPTIONS') {
    res.writeHead(200);
    res.end();
    return;
  }
  
  // Health check endpoint
  if (req.url === '/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'ok', allowedDirectory: currentAllowedDirectory }));
    return;
  }
  
  // Configure endpoint - POST /configure with {"allowedDirectory": "/path"}
  if (req.url === '/configure' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => {
      body += chunk.toString();
    });
    
    req.on('end', async () => {
      try {
        const config = JSON.parse(body);
        if (config.allowedDirectory) {
          // Wait for any ongoing initialization to complete
          while (isInitializing) {
            await new Promise(resolve => setTimeout(resolve, 100));
          }
          
          // Now start the new MCP process (this will wait until fully initialized)
          await startMCPProcess(config.allowedDirectory);
          console.log('MCP process configured and ready');
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ 
            status: 'configured', 
            allowedDirectory: currentAllowedDirectory 
          }));
        } else {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'allowedDirectory required' }));
        }
      } catch (error) {
        console.error('Failed to configure MCP process:', error);
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: error.message }));
      }
    });
    return;
  }
  
  
  // List tools endpoint - GET /tools
  if ((req.url === '/tools' || req.url === '/tools/') && req.method === 'GET') {
    try {
      if (!isReady) {
        res.writeHead(503, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'MCP server not ready yet, please wait' }));
        return;
      }
      const result = await sendMCPRequest('tools/list');
      
      // Filter out 'initialize_environment' from the tools list (it's a hidden server management tool)
      if (result && result.tools && Array.isArray(result.tools)) {
        result.tools = result.tools.filter(tool => tool.name !== 'initialize_environment');
      }
      
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(result));
    } catch (error) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: error.message }));
    }
    return;
  }
  
  // Call tool endpoint - POST /mcp/tools/{tool_name}
  if (req.url.startsWith('/mcp/tools/') && req.method === 'POST') {
    const toolName = req.url.substring(11); // Remove '/mcp/tools/'
    
    let body = '';
    req.on('data', chunk => {
      body += chunk.toString();
    });
    
    req.on('end', async () => {
      try {
        // Special handling for initialize_environment
        if (toolName === 'initialize_environment') {
          // Wait for any ongoing initialization to complete
          while (isInitializing) {
            await new Promise(resolve => setTimeout(resolve, 100));
          }
          
          console.log('Starting environment initialization...');
          
          // Step 1: List all subdirectories in /app/test_environments (including nested)
          const testEnvPath = '/app/test_environments';
          if (!fs.existsSync(testEnvPath)) {
            throw new Error(`Test environments directory not found: ${testEnvPath}`);
          }
          
          // Collect both top-level directories and their immediate subdirectories
          const subdirectories = [];
          const entries = fs.readdirSync(testEnvPath, { withFileTypes: true });
          
          // Helper function to check if nested directory has enough subdirectories (> 2)
          const hasEnoughSubdirectories = (dirPath) => {
            try {
              const entries = fs.readdirSync(dirPath, { withFileTypes: true });
              // Count subdirectories
              const subdirCount = entries.filter(entry => entry.isDirectory()).length;
              // Only include if more than 2 subdirectories
              return subdirCount > 2;
            } catch (e) {
              return false;
            }
          };
          
          for (const entry of entries) {
            if (entry.isDirectory()) {
              const topLevelPath = path.join(testEnvPath, entry.name);
              
              // Always add top-level directory (e.g., "desktop")
              subdirectories.push(entry.name);
              
              // Add nested subdirectories only if they contain > 2 subdirectories
              try {
                const subEntries = fs.readdirSync(topLevelPath, { withFileTypes: true });
                let nestedCount = 0;
                for (const subEntry of subEntries) {
                  if (subEntry.isDirectory()) {
                    const nestedFullPath = path.join(topLevelPath, subEntry.name);
                    // Only add if it has > 2 subdirectories
                    if (hasEnoughSubdirectories(nestedFullPath)) {
                      const relativePath = path.join(entry.name, subEntry.name);
                      subdirectories.push(relativePath);
                      nestedCount++;
                    }
                  }
                }
                if (nestedCount > 0) {
                  console.log(`  Added ${nestedCount} nested directories (with > 2 subdirs) from ${entry.name}`);
                }
              } catch (e) {
                console.log(`  Could not read subdirectories of ${entry.name}: ${e.message}`);
              }
            }
          }
          
          if (subdirectories.length === 0) {
            throw new Error('No subdirectories found in test_environments');
          }
          
          console.log(`Found ${subdirectories.length} candidate directories:`, subdirectories);
          
          // Step 2: Randomly select one subdirectory using cryptographically secure random
          const randomIndex = crypto.randomInt(0, subdirectories.length);
          const selectedSubdir = subdirectories[randomIndex];
          const sourcePath = path.join(testEnvPath, selectedSubdir);
          console.log(`Randomly selected: ${selectedSubdir}`);
          
          // Step 3: Generate random backup directory name with crypto random
          // Replace slashes with underscores for nested paths (e.g., "desktop/music" -> "desktop_music")
          const sanitizedSubdir = selectedSubdir.replace(/\//g, '_');
          const randomNum = crypto.randomInt(10000, 99999);
          const backupId = `backup_filesystem_${sanitizedSubdir}_${randomNum}`;
          const backupRoot = '/app/.mcpmark_backups';
          const backupRootWithId = path.join(backupRoot, backupId);
          
          // Create backup root directory if it doesn't exist
          if (!fs.existsSync(backupRoot)) {
            fs.mkdirSync(backupRoot, { recursive: true });
            console.log(`Created backup root directory: ${backupRoot}`);
          }
          
          // Step 4: Copy selected subdirectory contents to backup location
          const backupPath = backupRootWithId;
          console.log(`Copying ${sourcePath} to ${backupPath}...`);
          fs.cpSync(sourcePath, backupPath, { recursive: true });
          console.log('Copy completed successfully');
          
          // Step 5: Reconfigure MCP server with new backup directory
          console.log(`Reconfiguring MCP server with: ${backupPath}`);
          await startMCPProcess(backupPath);
          console.log('MCP server reconfigured and ready');
          
          // Step 6: Get directory tree using MCP tool
          console.log('Fetching directory tree...');
          const treeResult = await sendMCPRequest('tools/call', {
            name: 'directory_tree',
            arguments: { path: backupPath }
          });
          
          // Extract directory tree string from result (keep as formatted string)
          let directoryTree = '';
          let directoryTreeParsed = null;
          if (treeResult && treeResult.content) {
            for (const item of treeResult.content) {
              if (item.type === 'text') {
                // Keep as string (formatted JSON with newlines)
                directoryTree = item.text;
                try {
                  directoryTreeParsed = JSON.parse(item.text);
                } catch (e) {
                  console.log('Could not parse directory tree for sampling');
                }
                break;
              }
            }
          }
          
          // Step 7: Sample 2 random files from the directory tree
          console.log('Sampling random files...');
          
          // Helper function to check if file should be ignored
          const shouldIgnoreFile = (fileName) => {
            const lowerName = fileName.toLowerCase();
            // Ignore files starting with . (dot files)
            if (lowerName.startsWith('.')) {
              return true;
            }
            // Ignore files ending with .md (markdown files)
            if (lowerName.endsWith('.md')) {
              return true;
            }
            return false;
          };
          
          // Helper function to collect all file paths from directory tree
          const collectFilePaths = (tree, parentPath = '') => {
            const files = [];
            if (Array.isArray(tree)) {
              for (const item of tree) {
                if (item.type === 'file') {
                  // Skip ignored files
                  if (!shouldIgnoreFile(item.name)) {
                    files.push(path.join(parentPath, item.name));
                  }
                } else if (item.type === 'directory' && item.children) {
                  const nestedFiles = collectFilePaths(item.children, path.join(parentPath, item.name));
                  files.push(...nestedFiles);
                }
              }
            }
            return files;
          };
          
          let sampledFiles = [];
          if (directoryTreeParsed) {
            const allFiles = collectFilePaths(directoryTreeParsed);
            console.log(`Found ${allFiles.length} eligible files for sampling`);
            
            // Randomly sample 2 files
            if (allFiles.length > 0) {
              const numSamples = Math.min(2, allFiles.length);
              const shuffled = [...allFiles].sort(() => crypto.randomBytes(1)[0] - 128);
              const selectedFiles = shuffled.slice(0, numSamples);
              console.log(`Sampled files:`, selectedFiles);
              
              // Read content of each sampled file
              for (const filePath of selectedFiles) {
                try {
                  const fullPath = path.join(backupPath, filePath);
                  console.log(`Reading file: ${fullPath}`);
                  const fileResult = await sendMCPRequest('tools/call', {
                    name: 'read_file',
                    arguments: { path: fullPath }
                  });
                  
                  // Extract content from MCP result
                  let content = '';
                  if (fileResult && fileResult.content) {
                    for (const item of fileResult.content) {
                      if (item.type === 'text') {
                        content = item.text;
                        break;
                      }
                    }
                  }
                  
                  sampledFiles.push({
                    path: filePath,
                    content: content
                  });
                } catch (e) {
                  console.log(`Failed to read file ${filePath}: ${e.message}`);
                }
              }
            }
          }
          
          // Return response in simple, direct format
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({
            status: 'success',
            directory: backupPath,
            directory_tree: directoryTree,
            sampled_files: sampledFiles
          }));
          return;
        }
        
        // Normal MCP tool call
        if (!isReady) {
          res.writeHead(503, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'MCP server not ready yet, please wait' }));
          return;
        }
        const params = body ? JSON.parse(body) : {};
        const result = await sendMCPRequest('tools/call', {
          name: toolName,
          arguments: params
        });
        
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(result));
      } catch (error) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        // For initialize_environment errors, return simple format
        if (toolName === 'initialize_environment') {
          res.end(JSON.stringify({ 
            status: 'error',
            error: error.message,
            details: 'Failed to initialize environment. Please retry.'
          }));
        } else {
          // For normal MCP tool errors, use MCP format
          res.end(JSON.stringify({ 
            content: [{
              type: 'text',
              text: `Error: ${error.message}`
            }],
            isError: true
          }));
        }
      }
    });
    return;
  }
  
  // 404 for unknown endpoints
  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: 'Not found' }));
});

// Start server after MCP initialization
setTimeout(async () => {
  try {
    await initializeMCP();
    server.listen(port, '0.0.0.0', () => {
      console.log(`\n✓ Filesystem MCP REST Server running on http://0.0.0.0:${port}`);
      console.log(`\nAvailable endpoints:`);
      console.log(`  GET  /health                        - Health check`);
      console.log(`  GET  /tools                         - List available tools`);
      console.log(`  POST /mcp/tools/{name}              - Execute a tool`);
      console.log(`  POST /mcp/tools/initialize_environment - Initialize random isolated environment (hidden tool)`);
      console.log(`\nExamples:`);
      console.log(`  curl http://localhost:${port}/health`);
      console.log(`  curl http://localhost:${port}/tools`);
      console.log(`  curl -X POST http://localhost:${port}/mcp/tools/initialize_environment -d '{}'`);
      console.log(`  curl -X POST http://localhost:${port}/mcp/tools/read_file -d '{"path":"file.txt"}'\n`);
    });
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}, 1000);

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('Shutting down...');
  if (mcpProcess) mcpProcess.kill();
  server.close();
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('Shutting down...');
  if (mcpProcess) mcpProcess.kill();
  server.close();
  process.exit(0);
});

